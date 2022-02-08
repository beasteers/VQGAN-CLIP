# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun
import os
import glob
import sys
import itertools
import subprocess
from urllib.request import urlopen
from tqdm import tqdm
import numpy as np

from CLIP import clip
from PIL import ImageFile, Image, PngImagePlugin, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch import optim
from torchvision import transforms
from torchvision.transforms import functional as TF
# assign so we can just use getattr
from torch_optimizer import DiffGrad, AdamP
optim.DiffGrad, optim.AdamP = DiffGrad, AdamP

# pip install taming-transformers doesn't work with Gumbel, but does not yet work with coco etc
# appending the path does work with Gumbel, but gives ModuleNotFoundError: No module named 'transformers' for coco etc
sys.path.append('taming-transformers')
from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan

# NOTE: be wary of circular imports
from .cutout_util import *


normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
# From imagenet - Which is better?
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# create initial gradient image
def gradient_2d(start, stop, width, height, is_horizontal):
    return (
        np.tile(np.linspace(start, stop, width), (height, 1)) if is_horizontal else
        np.tile(np.linspace(start, stop, height), (width, 1)).T)

def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)
    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)
    return result


# NR: Testing with different intital images
def random_noise_image(w,h):
    return Image.fromarray(np.random.randint(0,255,(w,h,3),dtype=np.dtype('uint8')))

    
def random_gradient_image(w,h):
    return Image.fromarray(np.uint8(gradient_3d(
        w, h, (0, 0, np.random.randint(0,255)), 
        (np.random.randint(1,255), np.random.randint(2,255), np.random.randint(3,128)), 
        (True, False, False))))


def default_random_embedding(model, size, device):
    sideX, sideY, toksX, toksY = size_from_model(size, model)
    gumbel = is_gumbel(model)
    emb_layer = model.quantize.embed if gumbel else model.quantize.embedding

    e_dim = 256 if gumbel else model.quantize.e_dim
    n_toks = model.quantize.n_embed if gumbel else model.quantize.n_e
    one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
    return (one_hot @ emb_layer.weight).view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def is_gumbel(model):
    return isinstance(model, vqgan.GumbelVQ)


def split_items(xs, ch):
    return (xs.split(ch) if isinstance(xs, str) else xs or ())


class Promptor:
    def __init__(self, perceptor, device, make_cutouts, sides):
        self.perceptor = perceptor
        self.make_cutouts = make_cutouts
        self.device = device
        self.sides = sides

    def split(self, prompts, ch='|'):
        return [self.split_prompt(p) for p in split_items(prompts, ch)]

    #NR: Split prompts and weights
    def split_prompt(prompt):
        vals = prompt.rsplit(':', 2)
        vals = vals + ['', '1', '-inf'][len(vals):]
        return vals[0], float(vals[1]), float(vals[2])

    def text(self, prompt, weight=1, stop=float('-inf')):
        embed = self.perceptor.encode_text(clip.tokenize(prompt).to(self.device)).float()
        return Prompt(embed, weight, stop).to(self.device)

    def image(self, prompt, weight=1, stop=float('-inf')):
        img = resize_image(Image.open(prompt).convert('RGB'), self.sides)
        batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
        embed = self.perceptor.encode_image(normalize(batch)).float()
        return Prompt(embed, weight, stop).to(self.device)

    def noise(self, seed, weight=1):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=gen)
        return Prompt(embed, weight).to(self.device)

    def __call__(self, texts=None, images=None, noise=None, noise_weights=None):
        ps = []
        ps.update(self.text(*a) for a in self.split(texts))
        ps.update(self.image(*a) for a in self.split(images))
        ps.update(self.noise(seed, w) for seed, w in itertools.zip_longest(
            noise or (), noise_weights or (), fillvalue=1))
        return ps


def get_opt(z, name, lr, **kw):
    if name == "DiffGrad":
        kw = dict(dict(eps=1e-9, weight_decay=1e-9), **kw)
    return getattr(optim, name or 'Adam')([z], lr=lr)


def read_image(src, copy=False):
    img = Image.open(urlopen(src) if src.startswith('http') and '://' in src else src)
    if copy:
        img2 = img.copy()
        img.close()
        return img2
    return img


def resize_image(image, out_size):
    if not out_size:
        return image
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    return image.resize((round((area * ratio)**0.5), round((area / ratio)**0.5)), Image.LANCZOS)


# For zoom video
def zoom_at(img, x, y, zoom=1, shiftx=0, shifty=0):
    if zoom == 1:
        return img
    w, h = img.size
    img = (img.crop((
        x - w / 2 / zoom, y - h / 2 / zoom, 
        x + w / 2 / zoom, y + h / 2 / zoom
    )).resize((w, h), Image.LANCZOS))
    if shiftx or shifty:
        img = ImageChops.offset(img, shiftx, shifty)
    return img


def embed_image(model, src, sideX, sideY, device):
    # Load and resize image and Re-encode
    return embed_pil(model, resize_image(read_image(src).convert('RGB'), (sideX, sideY)), device)

def embed_pil(model, pil_image, device):
    z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    z_orig = z.clone()
    z.requires_grad_(True)
    return z, z_orig



def img2uint8(out):
    img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
    return np.array(np.transpose(img, (1, 2, 0)))


def size_from_model(size, model):
    f = 2**(model.decoder.num_resolutions - 1)
    toksX, toksY = size[0] // f, size[1] // f
    sideX, sideY = toksX * f, toksY * f
    return sideX, sideY, toksX, toksY


# Vector quantize
def synth(model, z):
    weight = model.quantize.embed.weight if is_gumbel(model) else model.quantize.embedding.weight
    z_q = vector_quantize(z.movedim(1, 3), weight).movedim(3, 1)
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


#@torch.no_grad()
@torch.inference_mode()
def checkin(model, z, i, losses, output=None, comment=None):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
    if output:
        out = synth(model, z)
        info = PngImagePlugin.PngInfo()
        if comment:
            info.add_text('comment', f'{comment}')
        TF.to_pil_image(out[0].cpu()).save(output, pnginfo=info) 	





def frames2video(frame_dir, video_file, fps=10):
    cmd = [
        'ffmpeg', '-y', '-i', f'{frame_dir}/frame%04d.jpg', 
        '-b:v', '8M', '-c:v', 'h264_nvenc', '-pix_fmt', 'yuv420p', '-strict', '-2', 
        '-filter:v', f'"minterpolate=\'mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps={fps}\'"', video_file]
    subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
    return video_file


def video2frames(video_file, output_dir):
    output_dir = output_dir or os.path.split(video_file)[0]
    cmd = ['ffmpeg', '-y', '-i', video_file, '-q:v', '2', f'{output_dir}/frame%04d.jpg']
    subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
    return glob.glob(os.path.join(output_dir, 'frame*.jpg'))


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