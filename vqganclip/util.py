# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

import argparse
import math
import random
# from email.policy import default
from urllib.request import urlopen
from tqdm import tqdm
import sys
import os

# pip install taming-transformers doesn't work with Gumbel, but does not yet work with coco etc
# appending the path does work with Gumbel, but gives ModuleNotFoundError: No module named 'transformers' for coco etc
sys.path.append('taming-transformers')

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
#import taming.modules 

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties
torch.backends.cudnn.benchmark = False		# NR: True is a bit faster, but can lead to OOM. False is more deterministic.
#torch.use_deterministic_algorithms(True)	# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation

from torch_optimizer import DiffGrad, AdamP

from CLIP import clip
import kornia.augmentation as K
import numpy as np
import imageio

from PIL import ImageFile, Image, PngImagePlugin, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True

from subprocess import Popen, PIPE
import re

# Supress warnings
import warnings
warnings.filterwarnings('ignore')

# NOTE: be wary of circular imports
from .cutout_util import *


normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
# From imagenet - Which is better?
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



# For zoom video
def zoom_at(img, x, y, zoom):
    if zoom == 1:
        return img
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2, 
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)



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
    array = gradient_3d(w, h, (0, 0, np.random.randint(0,255)), (np.random.randint(1,255), np.random.randint(2,255), np.random.randint(3,128)), (True, False, False))
    return Image.fromarray(np.uint8(array))



def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)



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


#NR: Split prompts and weights
def split_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


def split_items(x, sep='|'):
    return x.split(sep) if isinstance(x, str) else (x or [])


def load_prompts(perceptor, texts=None, images=None, noise_seeds=None, noise_weights=None, make_cutouts=None, sides=None, device=None):
    sideX, sideY = sides
    pMs = []
    # CLIP tokenize/encode
    texts = texts.split('|') if isinstance(texts, str) else texts or ()
    for prompt in texts:
        txt, weight, stop = split_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    images = images.split('|') if isinstance(images, str) else images or ()
    for prompt in images:
        path, weight, stop = split_prompt(prompt)
        img = Image.open(path)
        img = resize_image(img.convert('RGB'), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(noise_seeds or (), noise_weights or (1,)*len(noise_seeds or ())):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))

    return pMs




def get_opt(z, opt_name, opt_lr):
    if opt_name == "Adam":
        opt = optim.Adam([z], lr=opt_lr)	# LR=0.1 (Default)
    elif opt_name == "AdamW":
        opt = optim.AdamW([z], lr=opt_lr)	
    elif opt_name == "Adagrad":
        opt = optim.Adagrad([z], lr=opt_lr)	
    elif opt_name == "Adamax":
        opt = optim.Adamax([z], lr=opt_lr)	
    elif opt_name == "DiffGrad":
        opt = DiffGrad([z], lr=opt_lr, eps=1e-9, weight_decay=1e-9) # NR: Playing for reasons
    elif opt_name == "AdamP":
        opt = AdamP([z], lr=opt_lr)		    
    elif opt_name == "RAdam":
        opt = optim.RAdam([z], lr=opt_lr)		    
    elif opt_name == "RMSprop":
        opt = optim.RMSprop([z], lr=opt_lr)
    else:
        print("Unknown optimiser. Are choices broken?")
        opt = optim.Adam([z], lr=opt_lr)
    return opt


def reset_seed(seed=None):
    seed = torch.seed() if seed is None else seed
    torch.manual_seed(seed)
    print('Using seed:', seed)


def read_image(src, copy=False):
    img = Image.open(urlopen(src) if src.startswith('http') and '://' in src else src)
    if copy:
        img2 = img.copy()
        img.close()
        return img2
    return img


def embed_image(model, src, sideX, sideY, device):
    # Load and resize image and Re-encode
    img = Image.open(urlopen(src) if src.startswith('http') and '://' in src else src)
    pil_image = img.convert('RGB').resize((sideX, sideY), Image.LANCZOS)
    z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    z_orig = z.clone()
    z.requires_grad_(True)
    return z, z_orig


def img2uint8(out):
    img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
    img = np.transpose(img, (1, 2, 0))
    return np.array(img)


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
