#@title 2.2 Define necessary functions

import sys
sys.path.append('./SLIP')
sys.path.append('./ResizeRight')
from dataclasses import dataclass
from functools import partial
import gc
import io
import math
import timm
from IPython import display
import lpips
from PIL import Image, ImageOps
import requests
from glob import glob
import json
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
sys.path.append('./CLIP')
sys.path.append('./guided-diffusion')
import clip
from resize_right import resize
from models import SLIP_VITB16, SLIP, SLIP_VITL16
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from ipywidgets import Output
import hashlib

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if torch.cuda.get_device_capability(device) == (8,0): ## A100 fix thanks to Emad
  print('Disabling CUDNN for A100 gpu', file=sys.stderr)
  torch.backends.cudnn.enabled = False

torch.backends.cudnn.deterministic = True


def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        

# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869

def interp(t):
    return 3 * t**2 - 2 * t ** 3

def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)

def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0]//3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out

def regen_perlin():
    if perlin_mode == 'color':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
    elif perlin_mode == 'gray':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
    else:
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)

    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(batch_size, -1, -1, -1)


def perlin_init_image(perlin_mode):
    if perlin_mode == 'color':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
    elif perlin_mode == 'gray':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
    else:
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
    return TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)


def clip_prompts(clip_model, frame_prompt, fuzzy_prompt=False, rand_mag, fuzzy_count=25):
    target_embeds = []
    weights = []
    model_stat = {"clip_model":clip_model, "target_embeds":[], "make_cutouts":None, "weights":[]}
    # make_cutouts = MakeCutouts(clip_model.visual.input_resolution, cutn, skip_augs=skip_augs) 
    
    # for prompt in frame_prompt:
    txt, weight = parse_prompt(frame_prompt)
    txt = clip_model.encode_text(clip.tokenize(frame_prompt).to(device)).float()
    if fuzzy_prompt:
        target_embeds = [(txt + torch.randn(txt.shape).cuda() * rand_mag).clamp(0,1) for i in range(fuzzy_count)]
        weights = [weights]*fuzzy_count
    else:
        target_embeds = [txt]
        weights = [weight]

    # for prompt in image_prompts:
    #     path, weight = parse_prompt(prompt)
    #     img = Image.open(fetch(path)).convert('RGB')
    #     img = TF.resize(img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS)
    #     batch = model_stat["make_cutouts"](TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1))
    #     embed = clip_model.encode_image(normalize(batch)).float()
    #     if fuzzy_prompt:
    #         for i in range(25):
    #             model_stat["target_embeds"].append((embed + torch.randn(embed.shape).cuda() * rand_mag).clamp(0,1))
    #             weights.extend([weight / cutn] * cutn)
    #     else:
    #         model_stat["target_embeds"].append(embed)
    #         model_stat["weights"].extend([weight / cutn] * cutn)

    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    return {
        'clip_model': clip_model, 
        'target_embeds': target_embeds,
        'weights': weights / weights.sum().abs(),
        # 'make_cutouts': make_cutouts,
    }


def scaleFrame(input_file, output_file, translation_x, translation_y, angle, zoom):
    img_0 = cv2.imread(input_file)
    center = (1*img_0.shape[1]//2, 1*img_0.shape[0]//2)
    trans_mat = np.float32(
            [[1, 0, translation_x],
            [0, 1, translation_y]]
    )
    rot_mat = cv2.getRotationMatrix2D( center, angle, zoom )

    trans_mat = np.vstack([trans_mat, [0,0,1]])
    rot_mat = np.vstack([rot_mat, [0,0,1]])
    transformation_matrix = np.matmul(rot_mat, trans_mat)

    img_0 = cv2.warpPerspective(
            img_0,
            transformation_matrix,
            (img_0.shape[1], img_0.shape[0]),
            borderMode=cv2.BORDER_WRAP
    )
    cv2.imwrite(output_file, img_0)

    # zoom = zoom_per_frame
    # new_img = img.resize((math.floor(side_x*zoom),math.floor(side_y*zoom)))
    # w = math.floor(side_x*zoom)
    # h = math.floor(side_y*zoom)
    # cw = side_x
    # ch = side_y
    # new_img = new_img.rotate(rotation_per_frame)
    # box = w//2 - cw//2, h//2 - ch//2, w//2 + cw//2, h//2 + ch//2
    # # Crop the center of the image
    # new_img = new_img.crop(box)
    # new_img.save('prevFrameScaled.png')
    
    return output_file



def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def read_image_workaround(path):
    """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
    this incompatibility to avoid colour inversions."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()

def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]

def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2]//4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(cutn):
            if ch > cutn - cutn//4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(float(self.cut_size/max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts

cutout_debug = False
padargs = {}

class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size,
                 Overview=4, 
                 InnerCrop = 0, IC_Size_Pow=0.5, IC_Grey_P = 0.2
                 ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.4),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.1),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
            # T.RandomHorizontalFlip(p=0.5),
            # T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            # T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.RandomPerspective(distortion_scale=0.4, p=0.7),
            # T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.RandomGrayscale(p=0.15),
            # T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1,3,self.cut_size,self.cut_size] 
        output_shape_2 = [1,3,self.cut_size+2,self.cut_size+2]
        pad_input = F.pad(input,((sideY-max_size)//2,(sideY-max_size)//2,(sideX-max_size)//2,(sideX-max_size)//2), **padargs)
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview>0:
            if self.Overview<=4:
                if self.Overview>=1:
                    cutouts.append(cutout)
                if self.Overview>=2:
                    cutouts.append(gray(cutout))
                if self.Overview>=3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview==4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if cutout_debug:
                TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("/content/cutout_overview0.jpg",quality=99)
                              
        if self.InnerCrop >0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([])**self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("/content/cutout_InnerCrop.jpg",quality=99)
        cutouts = torch.cat(cutouts)
        if skip_augs is not True: cutouts=self.augs(cutouts)
        return cutouts

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)     

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete



def do_run(total_frames, key_frames, starting_init_scale, starting_skip_timesteps, starting_init_image):
    global seed

    angle_series = []
    zoom_series = []
    translation_x_series = []
    translation_y_series = []
    
    for frame_num in tqdm(range(total_frames)):
        if stop_on_next_loop:
            break
        if key_frames:
            angle = angle_series[frame_num]
            zoom = zoom_series[frame_num]
            translation_x = translation_x_series[frame_num]
            translation_y = translation_y_series[frame_num]
            print(f'angle: {angle} zoom: {zoom} translation_x: {translation_x} translation_y: {translation_y}')
        display.clear_output(wait=True)
        batchBar = tqdm(range(total_frames), desc ="Frames")
        batchBar.n = frame_num
        batchBar.refresh()
        init_scale = starting_init_scale
        skip_timesteps = starting_skip_timesteps
        init_image = starting_init_image or None

        if frame_num == 0:
            save_settings()
        if frame_num > 0:
            seed = seed + 1
            init_image = scaleFrame('progress.png', 'prevFrameScaled.png')
            init_scale = prev_frame_scale
            skip_timesteps = prev_frame_skip_timestep

        set_seed(seed)

        loss_values = []
        target_embeds, weights = [], []
        
        frame_prompt = prompts[-1 if frame_num >= len(prompts) else frame_num]
        print(f'Running: "{frame_prompt}"')

        model_stats = []
        for clip_model in clip_models:
            model_stats.append(clip_prompts(clip_model, frame_prompt, fuzzy_prompt, rand_mag, fuzzy_count))

        init = None
        if init_image is not None:
            init = util.read_image(init_image).convert('RGB').resize((side_x, side_y), Image.LANCZOS)
            init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
        if perlin_init:
            init = perlin_init_image(perlin_mode)

        cur_t = None

        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x_is_NaN = False
                x = x.detach().requires_grad_()
                n = x.shape[0]
                if use_secondary_model is True:
                    alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                    sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                    cosine_t = alpha_sigma_to_t(alpha, sigma)
                    out = secondary_model(x, cosine_t[None].repeat([n])).pred
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                else:
                    my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                    out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out['pred_xstart'] * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                for model_stat in model_stats:
                    for i in range(cutn_batches):
                        t_int = int(t.item())+1 #errors on last step without +1, need to find source
                        #when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                        try:
                            input_resolution=model_stat["clip_model"].visual.input_resolution
                        except:
                            input_resolution=224

                        cuts = MakeCutoutsDango(input_resolution,
                            Overview= cut_overview[1000-t_int], 
                            InnerCrop = cut_innercut[1000-t_int], IC_Size_Pow=cut_ic_pow, IC_Grey_P = cut_icgray_p[1000-t_int])
                        clip_in = normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                        dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                        dists = dists.view([cut_overview[1000-t_int]+cut_innercut[1000-t_int], n, -1])
                        losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                        loss_values.append(losses.sum().item()) # log loss, probably shouldn't do per cutn_batch
                        x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches
                tv_losses = tv_loss(x_in)
                range_losses = range_loss(out) if use_secondary_model is True else range_loss(out['pred_xstart'])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
                loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale

                if init is not None and init_scale:
                        init_losses = lpips_model(x_in, init)
                        loss = loss + init_losses.sum() * init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if torch.isnan(x_in_grad).any()==False:
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    # print("NaN'd")
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
            if clamp_grad and x_is_NaN == False:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=clamp_max) / magnitude  #min=-0.02, min=-clamp_max, 
            return grad

        ddim = model_config['timestep_respacing'].startswith('ddim')
        sample_fn = diffusion.ddim_sample_loop_progressive if ddim else diffusion.p_sample_loop_progressive

        image_display = Output()
        for i in range(n_batches):
                
            print('')
            display.display(image_display)
            gc.collect()
            torch.cuda.empty_cache()
            cur_t = diffusion.num_timesteps - skip_timesteps - 1
            total_steps = cur_t

            if perlin_init:
                init = regen_perlin()

            samples = sample_fn(
                model,
                (batch_size, 3, side_y, side_x),
                clip_denoised=clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_timesteps,
                init_image=init,
                randomize_class=randomize_class,
                **(dict(eta=eta) if ddim else {}))
            
            # with run_display:
            # display.clear_output(wait=True)
            for j, sample in enumerate(samples):    
                cur_t -= 1
                intermediateStep = (j % steps_per_checkpoint == 0 and j > 0 if steps_per_checkpoint is not None else j in intermediate_saves)
                # intermediateStep = False
                # if steps_per_checkpoint is not None:
                #     if j % steps_per_checkpoint == 0 and j > 0:
                #         intermediateStep = True
                # elif j in intermediate_saves:
                #     intermediateStep = True
                
                with image_display:
                    if j % display_rate == 0 or cur_t == -1 or intermediateStep:

                        for k, image in enumerate(sample['pred_xstart']):
                            # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                            current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
                            percent = math.ceil(j/total_steps*100)
                            if n_batches > 0:
                                #if intermediates are saved to the subfolder, don't append a step or percentage to the name
                                if cur_t == -1 and intermediates_in_subfolder is True:
                                    filename = f'{batch_name}({batchNum})_{frame_num:04}.png'
                                else:
                                    #If we're working with percentages, append it
                                    if steps_per_checkpoint is not None:
                                        filename = f'{batch_name}({batchNum})_{i:04}-{percent:02}%.png'
                                    # Or else, iIf we're working with specific steps, append those
                                    else:
                                        filename = f'{batch_name}({batchNum})_{i:04}-{j:03}.png'

                            image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                            image.save('progress.png')

                            if j % display_rate == 0 or cur_t == -1:
                                display.clear_output(wait=True)
                                display.display(display.Image('progress.png'))
                            if intermediateStep:
                                image.save(f'{partialFolder if intermediates_in_subfolder else batchFolder}/{filename}')
                            if cur_t == -1:
                                image.save('prevFrame.png')
                                image.save(f'{batchFolder}/{filename}')
                                display.clear_output()
            
            plt.plot(np.array(loss_values), 'r')



class KeyFrames:
    def __init__(self, *values):
        self._is_keyframes = [isinstance(v, (list, tuple)) for v in values]
        self.values = values

    def __call__(self, i):
        return [v[i] if isinstance(v, (list, tuple)) else v(i) if callable(v) else v for v in self.values]

class Disco:

    def init_image(self, init_image, perlin_init, perlin_mode, side_x, side_y):
        init = None
        if init_image is not None:
            init = util.read_image(init_image).convert('RGB').resize((side_x, side_y), Image.LANCZOS)
            init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
        if perlin_init:
            init = perlin_init_image(perlin_mode)


    starting_init_scale = None
    starting_skip_timesteps = None
    starting_init_image = None
    prev_frame_scale = None
    prev_frame_skip_timestep = None

    model = None
    clip_models = None
    side_x = None
    side_y = None
    # perlin_init
    # perlin_mode
    # angle_series
    # zoom_series
    # translation_x_series
    # translation_y_series


    # display
    steps_per_checkpoint = None
    intermediate_saves = None
    intermediates_in_subfolder = None
    display_rate = None
    total_steps = None
    partialFolder = None
    batchFolder = None

    def run(self, prompts, total_frames, n_batches, perlin_init, perlin_mode, 
            angle_series, zoom_series, translation_x_series, translation_y_series,
            fuzzy_prompt, rand_mag, fuzzy_count, clip_denoised, randomize_class, 
            batch_size, eta, clamp_grad):
        self.image_display = Output()
        display.display(self.image_display)

        kf = KeyFrames(angle_series, zoom_series, translation_x_series, translation_y_series)

        self.save_settings()
        for frame_num in tqdm(range(total_frames)):
            if stop_on_next_loop:
                break

            angle, zoom, translation_x, translation_y = kf(frame_num)
            print(f'angle: {angle} zoom: {zoom} translation_x: {translation_x} translation_y: {translation_y}')

            init_scale = self.starting_init_scale
            skip_timesteps = self.starting_skip_timesteps
            init_image = self.starting_init_image or None

            if frame_num > 0:
                init_image = scaleFrame('progress.png', 'prevFrameScaled.png')
                init_scale = self.prev_frame_scale
                skip_timesteps = self.prev_frame_skip_timestep

            set_seed(seed + frame_num)

            init = self.init_image(init_image, perlin_init, perlin_mode, self.side_x, self.side_y)
            frame_prompt = prompts[-1 if frame_num >= len(prompts) else frame_num]
            print(f'Running: "{frame_prompt}"')
            model_stats = [
                clip_prompts(clip_model, frame_prompt, fuzzy_prompt, rand_mag, fuzzy_count)
                for clip_model in self.clip_models
            ]
            
            cur_t = None

            ddim = model_config['timestep_respacing'].startswith('ddim')
        sample_fn = diffusion.ddim_sample_loop_progressive if ddim else diffusion.p_sample_loop_progressive

            loss_values = []
            for i in range(n_batches):
                gc.collect()
                torch.cuda.empty_cache()

                cur_t = diffusion.num_timesteps - skip_timesteps - 1
                total_steps = cur_t

                if perlin_init:
                    init = regen_perlin()

                samples = sample_fn(
                    self.model,
                    (batch_size, 3, self.side_y, self.side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=self.cond_fn(init, model_stats, loss_values, init_scale, clamp_grad),
                    progress=True,
                    skip_timesteps=skip_timesteps,
                    init_image=init,
                    randomize_class=randomize_class,
                    **(dict(eta=eta) if ddim else {}))
                for j, sample in enumerate(samples):
                    self.display(sample, frame_num, i, j, cur_t, batch_name, batchNum, n_batches, total_steps)

            plt.plot(np.array(loss_values), 'r')

    use_secondary_model = None
    secondary_model = None
    cutn_batches = None
    tv_scale = None
    range_scale = None
    sat_scale = None
    clip_guidance_scale = None

    def cond_fn(self, init, model_stats, cur_t, loss_values, init_scale, clamp_grad):
        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x_is_NaN = False
                x = x.detach().requires_grad_()
                n = x.shape[0]
                if use_secondary_model is True:
                    alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                    sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                    cosine_t = alpha_sigma_to_t(alpha, sigma)
                    out = self.secondary_model(x, cosine_t[None].repeat([n])).pred
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                else:
                    my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                    out = diffusion.p_mean_variance(self.model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out['pred_xstart'] * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                for model_stat in model_stats:
                    for i in range(cutn_batches):
                        t_int = int(t.item())+1 #errors on last step without +1, need to find source
                        #when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                        try:
                            input_resolution=model_stat["clip_model"].visual.input_resolution
                        except:
                            input_resolution=224

                        cuts = MakeCutoutsDango(input_resolution,
                            Overview= cut_overview[1000-t_int], 
                            InnerCrop = cut_innercut[1000-t_int], IC_Size_Pow=cut_ic_pow, IC_Grey_P = cut_icgray_p[1000-t_int])
                        clip_in = normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                        dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                        dists = dists.view([cut_overview[1000-t_int]+cut_innercut[1000-t_int], n, -1])
                        losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                        loss_values.append(losses.sum().item()) # log loss, probably shouldn't do per cutn_batch
                        x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches
                tv_losses = tv_loss(x_in)
                range_losses = range_loss(out) if use_secondary_model is True else range_loss(out['pred_xstart'])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
                loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale

                if init is not None and init_scale:
                        init_losses = lpips_model(x_in, init)
                        loss = loss + init_losses.sum() * init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if torch.isnan(x_in_grad).any()==False:
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    # print("NaN'd")
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
            if clamp_grad and x_is_NaN == False:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=clamp_max) / magnitude  #min=-0.02, min=-clamp_max, 
            return grad
        return cond_fn

    def display(self, sample, frame_num, i, j, cur_t, batch_name, batchNum, n_batches, total_steps):
        intermediateStep = (j % self.steps_per_checkpoint == 0 and j > 0 if self.steps_per_checkpoint else j in self.intermediate_saves)
        with self.image_display:
            if j % self.display_rate == 0 or cur_t == -1 or intermediateStep:
                for k, image in enumerate(sample['pred_xstart']):
                    if n_batches > 0:
                        current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
                        percent = math.ceil(j/total_steps*100)
                        fid = f'{frame_num:04}' if cur_t == -1 and self.intermediates_in_subfolder else f'{i:04}-{percent:02}%' if self.steps_per_checkpoint else f'{i:04}-{j:03}'
                        filename = f'{batch_name}({batchNum})_{fid}.png'

                    image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                    image.save('progress.png')

                    if j % self.display_rate == 0 or cur_t == -1:
                        display.clear_output(wait=True)
                        display.display(display.Image('progress.png'))
                    if intermediateStep:
                        image.save(f'{self.partialFolder if self.intermediates_in_subfolder else self.batchFolder}/{filename}')
                    if cur_t == -1:
                        image.save('prevFrame.png')
                        image.save(f'{self.batchFolder}/{filename}')
                        display.clear_output()


    def save_settings(self):
        setting_list = {
            'text_prompts': text_prompts,
            'image_prompts': image_prompts,
            'clip_guidance_scale': clip_guidance_scale,
            'tv_scale': tv_scale,
            'range_scale': range_scale,
            'sat_scale': sat_scale,
            # 'cutn': cutn,
            'cutn_batches': cutn_batches,
            'total_frames': total_frames,
            # 'rotation_per_frame': rotation_per_frame,
            'starting_init_image': starting_init_image,
            'starting_init_scale': starting_init_scale,
            'starting_skip_timesteps': starting_skip_timesteps,
            # 'zoom_per_frame': zoom_per_frame,
            'prev_frame_scale': prev_frame_scale,
            'prev_frame_skip_timestep': prev_frame_skip_timestep,
            'perlin_init': perlin_init,
            'perlin_mode': perlin_mode,
            'skip_augs': skip_augs,
            'randomize_class': randomize_class,
            'clip_denoised': clip_denoised,
            'clamp_grad': clamp_grad,
            'clamp_max': clamp_max,
            'seed': seed,
            'fuzzy_prompt': fuzzy_prompt,
            'rand_mag': rand_mag,
            'eta': eta,
            'width': width_height[0],
            'height': width_height[1],
            'diffusion_model': diffusion_model,
            'use_secondary_model': use_secondary_model,
            'steps': steps,
            'diffusion_steps': diffusion_steps,
            'ViTB32': ViTB32,
            'ViTB16': ViTB16,
            'RN101': RN101,
            'RN50': RN50,
            'RN50x4': RN50x4,
            'RN50x16': RN50x16,
            'cut_overview': str(cut_overview),
            'cut_innercut': str(cut_innercut),
            'cut_ic_pow': cut_ic_pow,
            'cut_icgray_p': str(cut_icgray_p),
            'key_frames': key_frames,
            'max_frames': max_frames,
            'angle': angle,
            'zoom': zoom,
            'translation_x': translation_x,
            'translation_y': translation_y,
        }
        # print('Settings:', setting_list)
        with open(f"{batchFolder}/{batch_name}({batchNum})_settings.txt", "w+") as f:   #save settings
            json.dump(setting_list, f, ensure_ascii=False, indent=4)
  




def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SecondaryDiffusionImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        self.timestep_embed = FourierFeatures(1, 16)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, c),
            ConvBlock(c, c),
            SkipBlock([
                nn.AvgPool2d(2),
                ConvBlock(c, c * 2),
                ConvBlock(c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool2d(2),
                    ConvBlock(c * 2, c * 4),
                    ConvBlock(c * 4, c * 4),
                    SkipBlock([
                        nn.AvgPool2d(2),
                        ConvBlock(c * 4, c * 8),
                        ConvBlock(c * 8, c * 4),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ]),
                    ConvBlock(c * 8, c * 4),
                    ConvBlock(c * 4, c * 2),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ]),
                ConvBlock(c * 4, c * 2),
                ConvBlock(c * 2, c),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ]),
            ConvBlock(c * 2, c),
            nn.Conv2d(c, 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


class SecondaryDiffusionImageNet2(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock([
                self.down,
                ConvBlock(cs[0], cs[1]),
                ConvBlock(cs[1], cs[1]),
                SkipBlock([
                    self.down,
                    ConvBlock(cs[1], cs[2]),
                    ConvBlock(cs[2], cs[2]),
                    SkipBlock([
                        self.down,
                        ConvBlock(cs[2], cs[3]),
                        ConvBlock(cs[3], cs[3]),
                        SkipBlock([
                            self.down,
                            ConvBlock(cs[3], cs[4]),
                            ConvBlock(cs[4], cs[4]),
                            SkipBlock([
                                self.down,
                                ConvBlock(cs[4], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[4]),
                                self.up,
                            ]),
                            ConvBlock(cs[4] * 2, cs[4]),
                            ConvBlock(cs[4], cs[3]),
                            self.up,
                        ]),
                        ConvBlock(cs[3] * 2, cs[3]),
                        ConvBlock(cs[3], cs[2]),
                        self.up,
                    ]),
                    ConvBlock(cs[2] * 2, cs[2]),
                    ConvBlock(cs[2], cs[1]),
                    self.up,
                ]),
                ConvBlock(cs[1] * 2, cs[1]),
                ConvBlock(cs[1], cs[0]),
                self.up,
            ]),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)
