# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun
import os
import sys
import glob
import time
from subprocess import Popen, PIPE
import numpy as np
from tqdm import tqdm
import imageio
from CLIP import clip
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision.transforms import functional as TF
torch.backends.cudnn.benchmark = False		# NR: True is a bit faster, but can lead to OOM. False is more deterministic.
#torch.use_deterministic_algorithms(True)	# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation

# pip install taming-transformers doesn't work with Gumbel, but does not yet work with coco etc
# appending the path does work with Gumbel, but gives ModuleNotFoundError: No module named 'transformers' for coco etc
sys.path.append('taming-transformers')

import warnings
warnings.filterwarnings('ignore')

from .util import *

# Check for GPU and reduce the default image size if low VRAM: # no gpu, <8gb (2 ** 33) ram, >8gb ram
default_image_size = (
    256 if not torch.cuda.is_available() else 
    304 if torch.cuda.get_device_properties(0).total_memory <= (2 ** 33) else 
    512)
img_exts = ['.jpg', '.png']

jit = "1.7.1" in torch.__version__



def initialize_z(model, init_image=None, init_noise=None, size=None, device=None):
    if init_image:
        img = read_image(init_image)
    elif init_noise == 'pixels':
        img = random_noise_image(size[0], size[1])
    elif init_noise == 'gradient':
        img = random_gradient_image(size[0], size[1])
    else:
        return default_random_embedding(model, size, device)

    sideX, sideY, _, _ = size_from_model(size, model)
    emb = model.quantize.embed if is_gumbel(model) else model.quantize.embedding
    z_min = emb.weight.min(dim=0).values[None, :, None, None]
    z_max = emb.weight.max(dim=0).values[None, :, None, None]
    z, z_orig = embed_pil(model, img.convert('RGB').resize((sideX, sideY), Image.LANCZOS), device)
    return z, z_orig, z_min, z_max



# Do it
def train_images(
        model, perceptor, prompts=None, image_prompts=(), 
        noise_seeds=(), noise_weights=(), video_style_dir=None, 
        init_image=None, init_noise=None, init_weight=0,
        zoom_video=False, zoom_scale=1, zoom_start=0, zoom_frequency=10, zoom_shift_x=0, zoom_shift_y=0, 
        prompt_frequency=0, max_iterations=500, make_video=False, 
        optimiser='Adam', step_size=0.1, seed=None, display_freq=50,
        cut_method='latest', cutn=32, cut_pow=1, size=(default_image_size, default_image_size), augments=None,
        outputs_dir='outputs', name=None, device=None):
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device) if isinstance(device, str) else device

    # get cutout function. Cutout class options: 'latest','original','updated' or 'updatedpooling'
    sideX, sideY, _, _ = size_from_model(size, model)
    make_cutouts = get_cutout_object(
        cut_method, perceptor.visual.input_resolution, cutn, cut_pow, 
        augments or ['Af', 'Pe', 'Ji', 'Er'])

    os.makedirs(outputs_dir, exist_ok=True)

    def run(init_image=None, init_noise=None, name='output', max_iterations=max_iterations):
        seed and torch.manual_seed(seed)
        print('Using seed:', torch.seed())
        
        z, z_orig, z_min, z_max = initialize_z(
            model, init_image, init_noise, size, device)

        out_dir = os.path.join(outputs_dir, name)
        debug_output_file = os.path.join(out_dir, 'debug.png')
        steps_dir = os.path.join(out_dir, 'steps')

        print('Using steps dir:', steps_dir)
        print('debug output with:', debug_output_file)

        # Make video steps directory
        if make_video or zoom_video:
            os.makedirs(steps_dir, exist_ok=True)
        j = p = 0 # Zoom video frame counter, phrase counter
        promptgen = Promptor(perceptor, device, make_cutouts, (sideX, sideY))
        pMs_static = promptgen(images=image_prompts, noise_seeds=noise_seeds, noise_weights=noise_weights)
        pMs_cycle = promptgen(all_phrases)
        pMs = [*pMs_cycle[:1], *pMs_static]

        opt = get_opt(z, optimiser, step_size)
        with tqdm(range(max_iterations)) as pbar:
            for i in pbar:
                # Change generated image
                if zoom_video and i % zoom_frequency == 0:
                    # img = img2uint8(synth(model, z))
                    # imageio.imwrite(os.path.join(steps_dir, f'zoom{j}.png'), img)  # Save image
                    # if zoom_start <= i:  # Time to start zooming
                    #     pil_image = zoom_at(
                    #         Image.fromarray(np.array(img).astype('uint8'), 'RGB'), 
                    #         sideX/2, sideY/2, zoom_scale, zoom_shift_x, zoom_shift_y)
                    #     z, z_orig = embed_pil(model, pil_image, device)
                    #     opt = get_opt(z, optimiser, step_size)
                    img = synth(model, z)
                    imageio.imwrite(os.path.join(steps_dir, f'zoom{j}.png'), img2uint8(img))  # Save image
                    if zoom_start <= i:  # Time to start zooming
                        pil_image = zoom_at(img, sideX/2, sideY/2, zoom_scale, zoom_shift_x, zoom_shift_y)
                        z, z_orig = embed_pil(model, pil_image, device)
                        opt = get_opt(z, optimiser, step_size)
                    j += 1

                # Change text prompt
                if prompt_frequency > 0 and pMs_cycle and i % prompt_frequency == 0 and i > 0:
                    p = (p + 1) % len(pMs_cycle)
                    tqdm.write(f'rotating to prompt {p}: {all_phrases[p]}')  # Show user we're changing prompt
                    pMs = [pMs_cycle[p], *pMs_static]
                
                # Training time

                opt.zero_grad(set_to_none=True)
                out = synth(model, z)
                iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
                
                # gather losses
                lossAll = [prompt(iii) for prompt in pMs]
                if init_weight:
                    lossAll.append(F.mse_loss(z, torch.zeros_like(z_orig)) * ((1/torch.tensor(i*2 + 1))*init_weight) / 2)

                # dump some training info
                if i % display_freq == 0:
                    checkin(model, z, i, lossAll, debug_output_file, prompts)

                # optimize the loss
                sum(lossAll).backward()
                opt.step()
                with torch.inference_mode():
                    z.copy_(z.maximum(z_min).minimum(z_max))

                # save image steps
                if make_video:
                    imageio.imwrite(os.path.join(steps_dir, f'step{i}.png'), img2uint8(out))

    # load prompt embeddings
    if not prompts and not image_prompts:
        prompts = "A cute, smiling, Nerdy Rodent"
    all_phrases = [phrase.strip().split("|") for phrase in split_items(prompts, '^')]
    image_prompts = [src.strip() for src in split_items(image_prompts, '|')]

    # load video styles, if provided
    init_styles = ([init_image] if init_image else []) + ([
        entry.path for entry in os.scandir(video_style_dir)
        if any(entry.path.endswith(e) for e in img_exts) and entry.is_file()
    ] if video_style_dir else [])

    # Output for the user
    print('Using device:', device)
    if prompts:
        print('Using text prompts:', prompts)  
    if image_prompts:
        print('Using image prompts:', image_prompts)
    print('Using initial styles:', init_styles or '[noise]')
    if noise_weights:
        print('Noise prompt weights:', noise_weights)

    output_dirs = []

    try:
        # if no styles were provided, use init image or noise
        if not init_styles:
            name_i = name or f'noise-{int(time.time())}'
            output_dirs.append(os.path.join(outputs_dir, name_i, 'steps'))
            run(None, init_noise, name_i)

        # run for each style
        for init_image in init_styles:
            name_i = name or os.path.basename(init_image).rsplit('.', 1)[0]
            output_dirs.append(os.path.join(outputs_dir, name_i, 'steps'))
            run(init_image, init_noise, name_i)
    except KeyboardInterrupt:
        print('Interrupted.')
    return output_dirs



def generate_video(img_dir, length=10, fps=0, input_fps=0, output_file=None, output_dir='outputs', zoom_video=False, comment=None, min_fps=10, max_fps=60):
    print(f'Generating video from {img_dir}...')
    # load the video frames
    fs = glob.glob(os.path.join(img_dir, f'{"zoom" if zoom_video else "step"}*.png'))
    if not fs:
        print(f'No images in {img_dir}')
        return
    frames = [Image.open(f) for f in fs]
    try:
        output_file = output_file or os.path.join(output_dir, f'{os.path.basename(output_dir).strip("/")}.mp4')
        if fps >= min_fps:
            ffmpeg_filter = f"minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps={fps}'"
            cmd = ['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(input_fps or fps), '-i', '-']
            cmd += ['-b:v', '10M', '-vcodec', 'h264_nvenc', '-pix_fmt', 'yuv420p', '-strict', '-2', '-filter:v', ffmpeg_filter]
        else:
            fps = np.clip(len(fs) / length, min_fps, max_fps)
            cmd = ['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-']
            cmd += ['-vcodec', 'libx264', '-r', str(fps), '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryslow']
        
        # feed the images to ffmpeg
        p = Popen(cmd + (['-metadata', f'comment={comment}']*bool(comment)) + [output_file], stdin=PIPE)        
    except FileNotFoundError:
        print("ffmpeg command failed - check your installation")
    for f in tqdm(frames):
        f.save(p.stdin, 'PNG')
    p.stdin.close()
    p.wait()


def main():
    args = parse_args()

    # Do it
    device = torch.device(args.cuda_device)
    model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
    perceptor = clip.load(args.clip_model, jit=float(torch.__version__[:3]) < 1.8)[0].eval().requires_grad_(False).to(device)

    # try:
    # import pyinstrument
    # pf = pyinstrument.Profiler()
    # with pf:
    out_dirs = train_images(
        model, perceptor, 
        prompts=args.prompts, image_prompts=args.image_prompts, 
        noise_seeds=args.noise_prompt_seeds, noise_weights=args.noise_prompt_weights, 
        video_style_dir=args.video_style_dir, 
        init_image=args.init_image, 
        init_noise=args.init_noise, 
        init_weight=args.init_weight,
        zoom_video=args.make_zoom_video, 
        zoom_scale=args.zoom_scale, zoom_start=args.zoom_start, 
        zoom_frequency=args.zoom_frequency, 
        zoom_shift_x=args.zoom_shift_x, zoom_shift_y=args.zoom_shift_y, 
        prompt_frequency=args.prompt_frequency, 
        max_iterations=args.max_iterations, 
        make_video=args.make_video, 
        name=args.name,
        optimiser=args.optimiser, step_size=args.step_size, 
        cut_method=args.cut_method, cutn=args.cutn, cut_pow=args.cut_pow, 
        size=args.size, augments=args.augments,
        seed=args.seed, device=device)
    # pf.print()

    # All done :)
    if args.make_video or args.make_zoom_video:
        for d in out_dirs:
            generate_video(
                d, args.video_length, output_dir=os.path.dirname(d), 
                fps=args.output_video_fps, input_fps=args.input_video_fps, 
                zoom_video=args.make_zoom_video,
                comment=args.prompts)
    # except KeyboardInterrupt:
    #     print('Interrupted.')
        # if args.make_video or args.make_zoom_video:
        #     if input("Still generate a video? y/[N] ").lower() != 'y':
        #         args.make_video = args.make_zoom_video = False
    # finally:


# Create the parser
def parse_args():
    import argparse
    vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')
    vq_parser.add_argument("-p",    "--prompts", type=str, help="Text prompts", default=None, dest='prompts')
    vq_parser.add_argument("-ip",   "--image_prompts", type=str, help="Image prompts / target image", default=[], dest='image_prompts')
    vq_parser.add_argument("-i",    "--iterations", type=int, help="Number of iterations", default=500, dest='max_iterations')
    vq_parser.add_argument("-se",   "--save_every", type=int, help="Save image iterations", default=50, dest='display_freq')
    vq_parser.add_argument("-s",    "--size", nargs=2, type=int, help="Image size (width height) (default: %(default)s)", default=[default_image_size,default_image_size], dest='size')
    vq_parser.add_argument("-ii",   "--init_image", type=str, help="Initial image", default=None, dest='init_image')
    vq_parser.add_argument("-in",   "--init_noise", type=str, help="Initial noise image (pixels or gradient)", default=None, dest='init_noise')
    vq_parser.add_argument("-iw",   "--init_weight", type=float, help="Initial weight", default=0., dest='init_weight')
    vq_parser.add_argument("-m",    "--clip_model", type=str, help="CLIP model (e.g. ViT-B/32, ViT-B/16)", default='ViT-B/32', dest='clip_model')
    vq_parser.add_argument("-conf", "--vqgan_config", type=str, help="VQGAN config", default=f'checkpoints/vqgan_imagenet_f16_16384.yaml', dest='vqgan_config')
    vq_parser.add_argument("-ckpt", "--vqgan_checkpoint", type=str, help="VQGAN checkpoint", default=f'checkpoints/vqgan_imagenet_f16_16384.ckpt', dest='vqgan_checkpoint')
    vq_parser.add_argument("-nps",  "--noise_prompt_seeds", nargs="*", type=int, help="Noise prompt seeds", default=[], dest='noise_prompt_seeds')
    vq_parser.add_argument("-npw",  "--noise_prompt_weights", nargs="*", type=float, help="Noise prompt weights", default=[], dest='noise_prompt_weights')
    vq_parser.add_argument("-lr",   "--learning_rate", type=float, help="Learning rate", default=0.1, dest='step_size')
    vq_parser.add_argument("-cutm", "--cut_method", type=str, help="Cut method", choices=['original','updated','nrupdated','updatedpooling','latest'], default='latest', dest='cut_method')
    vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=32, dest='cutn')
    vq_parser.add_argument("-cutp", "--cut_power", type=float, help="Cut power", default=1., dest='cut_pow')
    vq_parser.add_argument("-sd",   "--seed", type=int, help="Seed", default=None, dest='seed')
    vq_parser.add_argument("-opt",  "--optimiser", type=str, help="Optimiser", choices=['Adam','AdamW','Adagrad','Adamax','DiffGrad','AdamP','RAdam','RMSprop'], default='Adam', dest='optimiser')
    vq_parser.add_argument("-o",    "--output", type=str, help="Output filename", default=None, dest='output')
    vq_parser.add_argument("-n",    "--name", type=str, help="Output name", default=None, dest='name')
    vq_parser.add_argument("-vid",  "--video", action='store_true', help="Create video frames?", dest='make_video')
    vq_parser.add_argument("-zvid", "--zoom_video", action='store_true', help="Create zoom video?", dest='make_zoom_video')
    vq_parser.add_argument("-zs",   "--zoom_start", type=int, help="Zoom start iteration", default=0, dest='zoom_start')
    vq_parser.add_argument("-zse",  "--zoom_save_every", type=int, help="Save zoom image iterations", default=10, dest='zoom_frequency')
    vq_parser.add_argument("-zsc",  "--zoom_scale", type=float, help="Zoom scale %", default=0.99, dest='zoom_scale')
    vq_parser.add_argument("-zsx",  "--zoom_shift_x", type=int, help="Zoom shift x (left/right) amount in pixels", default=0, dest='zoom_shift_x')
    vq_parser.add_argument("-zsy",  "--zoom_shift_y", type=int, help="Zoom shift y (up/down) amount in pixels", default=0, dest='zoom_shift_y')
    vq_parser.add_argument("-cpe",  "--change_prompt_every", type=int, help="Prompt change frequency", default=10, dest='prompt_frequency')
    vq_parser.add_argument("-vl",   "--video_length", type=float, help="Video length in seconds (not interpolated)", default=10, dest='video_length')
    vq_parser.add_argument("-ofps", "--output_video_fps", type=float, help="Create an interpolated video (Nvidia GPU only) with this fps (min 10. best set to 30 or 60)", default=0, dest='output_video_fps')
    vq_parser.add_argument("-ifps", "--input_video_fps", type=float, help="When creating an interpolated video, use this as the input fps to interpolate from (>0 & <ofps)", default=15, dest='input_video_fps')
    vq_parser.add_argument("-d",    "--deterministic", action='store_true', help="Enable cudnn.deterministic?", dest='cudnn_determinism')
    vq_parser.add_argument("-aug",  "--augments", nargs='+', action='append', type=str, choices=['Ji','Sh','Gn','Pe','Ro','Af','Et','Ts','Cr','Er','Re'], help="Enabled augments (latest vut method only)", default=[], dest='augments')
    vq_parser.add_argument("-vsd",  "--video_style_dir", type=str, help="Directory with video frames to style", default=None, dest='video_style_dir')
    vq_parser.add_argument("-cd",   "--cuda_device", type=str, help="Cuda device to use", default="cuda:0", dest='cuda_device')
    args = vq_parser.parse_args()

    if not args.prompts and not args.image_prompts:
        args.prompts = "A cute, smiling, Nerdy Rodent"

    if args.cudnn_determinism:
        torch.backends.cudnn.deterministic = True

    if args.make_video and args.make_zoom_video:
        args.make_video = False

    # Fallback to CPU if CUDA is not found and make sure GPU video rendering is also disabled
    # NB. May not work for AMD cards?
    if args.cuda_device != 'cpu' and not torch.cuda.is_available():
        args.cuda_device = 'cpu'
        args.video_fps = 0
        print("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
        print("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")
    return args




if __name__ == '__main__':
    main()
