# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun
import os
import sys
import glob
import argparse
import numpy as np
from tqdm import tqdm

# pip install taming-transformers doesn't work with Gumbel, but does not yet work with coco etc
# appending the path does work with Gumbel, but gives ModuleNotFoundError: No module named 'transformers' for coco etc
sys.path.append('taming-transformers')

import torch
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties
torch.backends.cudnn.benchmark = False		# NR: True is a bit faster, but can lead to OOM. False is more deterministic.
#torch.use_deterministic_algorithms(True)	# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation

import imageio
from CLIP import clip
from PIL import ImageFile, Image, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Supress warnings
import warnings
warnings.filterwarnings('ignore')

from .util import *

# Check for GPU and reduce the default image size if low VRAM
default_image_size = 512  # >8GB VRAM
if not torch.cuda.is_available():
    default_image_size = 256  # no GPU found
elif get_device_properties(0).total_memory <= 2 ** 33:  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
    default_image_size = 318  # <8GB VRAM


# Create the parser
def parse_args():
    
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




def initialize_z(model, init_image=None, init_noise=None, size=None, device=None):
    sideX, sideY, toksX, toksY = size_from_model(size, model)
    gumbel = is_gumbel(model)
    emb_layer = model.quantize.embed if gumbel else model.quantize.embedding
    z_min = emb_layer.weight.min(dim=0).values[None, :, None, None]
    z_max = emb_layer.weight.max(dim=0).values[None, :, None, None]

    if init_image:
        img = Image.open(urlopen(init_image) if init_image.startswith('http') else init_image)
    elif init_noise == 'pixels':
        img = random_noise_image(size[0], size[1])
    elif init_noise == 'gradient':
        img = random_gradient_image(size[0], size[1])
    else:
        e_dim = 256 if gumbel else model.quantize.e_dim
        n_toks = model.quantize.n_embed if gumbel else model.quantize.n_e
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        return (one_hot @ emb_layer.weight).view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 

    pil_image = img.convert('RGB').resize((sideX, sideY), Image.LANCZOS)
    z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    z_orig = z.clone()
    z.requires_grad_(True)
    return z, z_orig, z_min, z_max

# Do it
img_exts = ['.jpg', '.png']
def train_images(
        model, perceptor, prompts=None, image_prompts=(), noise_seeds=(), noise_weights=(), video_style_dir=None, 
        init_image=None, init_noise=None, init_weight=0,
        zoom_video=False, zoom_scale=1, zoom_start=0, zoom_frequency=10, zoom_shift_x=0, zoom_shift_y=0, 
        prompt_frequency=0, max_iterations=500, make_video=False, 
        steps_dir='steps', outputs_dir='outputs', debug_output_file=None, name=None,
        optimiser='Adam', step_size=0.1, seed=None, display_freq=50,
        cut_method='latest', cutn=32, cut_pow=1, size=(default_image_size, default_image_size), augments=None,
        device=None):
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, str):
        device = torch.device(device)

    # Cutout class options: 'latest','original','updated' or 'updatedpooling'
    sideX, sideY, _, _ = size_from_model(size, model)

    # get cutout function. Cutout class options: 'latest','original','updated' or 'updatedpooling'
    cut_size = perceptor.visual.input_resolution
    augments = augments or ['Af', 'Pe', 'Ji', 'Er']
    make_cutouts = get_cutout_object(cut_method, cut_size, cutn, cut_pow, augments)

    os.makedirs(outputs_dir, exist_ok=True)

    def run(z, z_orig, steps_dir='steps', debug_output_file=None, max_iterations=max_iterations):
        print('Using steps dir:', steps_dir)
        if debug_output_file:
            print('debug output with:', debug_output_file)

        # Make video steps directory
        if make_video or zoom_video:
            os.makedirs(steps_dir, exist_ok=True)
        j = 0 # Zoom video frame counter
        p = 1 # Phrase counter
        pMs = load_prompts(
            perceptor, texts=all_phrases and all_phrases[0], images=image_prompts, 
            noise_seeds=noise_seeds, noise_weights=noise_weights, 
            make_cutouts=make_cutouts, sides=(sideX, sideY))

        opt = get_opt(z, optimiser, step_size)
        with tqdm(range(max_iterations)) as pbar:
            for i in pbar:
                # Change generated image
                if zoom_video and i % zoom_frequency == 0:
                    img = img2uint8(synth(model, z))
                    imageio.imwrite(os.path.join(steps_dir, f'zoom{j}.png'), img)  # Save image

                    # Time to start zooming?                    
                    if zoom_start <= i:
                        pil_image = Image.fromarray(np.array(img).astype('uint8'), 'RGB')
                        pil_image = zoom_at(pil_image, sideX/2, sideY/2, zoom_scale)
                        if zoom_shift_x or zoom_shift_y:
                            pil_image = ImageChops.offset(pil_image, zoom_shift_x, zoom_shift_y)

                        # Re-encode
                        z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
                        z_orig = z.clone()
                        z.requires_grad_(True)
                    j += 1

                # Change text prompt
                if prompt_frequency > 0 and i % prompt_frequency == 0 and i > 0:
                    if len(all_phrases) > 1:
                        p = p % len(all_phrases)  # loop
                        tqdm.write(f'rotating to prompt {p}: {all_phrases[p]}')  # Show user we're changing prompt
                        pMs = load_prompts(
                            perceptor, texts=all_phrases[p], images=image_prompts, 
                            noise_seeds=noise_seeds, noise_weights=noise_weights,
                            make_cutouts=make_cutouts, sides=(sideX, sideY))
                        p += 1
                
                # Training time

                opt.zero_grad(set_to_none=True)
                out = synth(model, z)
                iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
                
                # gather losses
                lossAll = []
                if init_weight:
                    lossAll.append(F.mse_loss(z, torch.zeros_like(z_orig)) * ((1/torch.tensor(i*2 + 1))*init_weight) / 2)
                lossAll.extend(prompt(iii) for prompt in pMs)

                # dump some training info
                if i % display_freq == 0:
                    checkin(model, z, i, lossAll, debug_output_file, prompts)

                # optimize the loss
                loss = sum(lossAll)
                loss.backward()
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
    init_styles = [
        entry.path for entry in os.scandir(video_style_dir)
        if any(entry.path.endswith(e) for e in img_exts) and entry.is_file()
    ] if video_style_dir else []

    # Output for the user
    print('Using device:', device)
    if prompts:
        print('Using text prompts:', prompts)  
    if image_prompts:
        print('Using image prompts:', image_prompts)
    if init_image:
        print('Using initial image:', init_image)
    if noise_weights:
        print('Noise prompt weights:', noise_weights)

    output_dirs = []

    try:
        # if no styles were provided, use init image or noise
        if not init_styles:
            reset_seed(seed)
            z, z_orig, z_min, z_max = initialize_z(model, init_image, init_noise, size, device)
            # get output paths
            name_i = (
                name or (init_image and os.path.basename(init_image).rsplit('.', 1)[0]) or 
                (debug_output_file and os.path.basename(debug_output_file).rsplit('.', 1)[0]) or 'output')
            out_dir = os.path.join(outputs_dir, name_i)
            debug_output_file = debug_output_file or os.path.join(out_dir, f'{name_i}.png')
            steps_dir_i = os.path.join(out_dir, 'steps')
            output_dirs.append(steps_dir_i)
            # optimize image
            run(z, z_orig, steps_dir_i, debug_output_file)

        # run for each style
        for init_image in init_styles:
            reset_seed(seed)
            z, z_orig = embed_image(model, init_image, sideX, sideY, device)
            # get output paths
            name_i = name or os.path.basename(init_image).rsplit('.', 1)[0]
            # debug_output_file = os.path.join(outputs_dir, f'{name_i}.png')
            # steps_dir_i = os.path.join(steps_dir, name_i)
            out_dir = os.path.join(outputs_dir, name_i)
            debug_output_file = debug_output_file or os.path.join(out_dir, f'{name_i}.png')
            steps_dir_i = os.path.join(out_dir, 'steps')
            output_dirs.append(steps_dir_i)
            # optimize image
            run(z, z_orig, steps_dir_i, debug_output_file)
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
        output_file = output_file or os.path.join(output_dir, f'{os.path.basename(output_dir)}.mp4')
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
        video_style_dir=args.video_style_dir, init_image=args.init_image, init_noise=args.init_noise, init_weight=args.init_weight,
        zoom_video=args.make_zoom_video, zoom_scale=args.zoom_scale, zoom_start=args.zoom_start, 
        zoom_frequency=args.zoom_frequency, zoom_shift_x=args.zoom_shift_x, zoom_shift_y=args.zoom_shift_y, 
        prompt_frequency=args.prompt_frequency, max_iterations=args.max_iterations, make_video=args.make_video, 
        debug_output_file=args.output, name=args.name,
        optimiser=args.optimiser, step_size=args.step_size, seed=args.seed,
        cut_method=args.cut_method, cutn=args.cutn, cut_pow=args.cut_pow, 
        size=args.size, augments=args.augments,
        device=device)
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
        


if __name__ == '__main__':
    main()
