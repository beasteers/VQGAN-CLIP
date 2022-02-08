import os
import time
import torch
import imageio
from .util import *

model_dir = os.path.abspath(__file__, '../../checkpoints')

# Check for GPU and reduce the default image size if low VRAM: # no gpu, <8gb (2 ** 33) ram, >8gb ram
default_image_size = (
    256 if not torch.cuda.is_available() else 
    304 if torch.cuda.get_device_properties(0).total_memory <= (2 ** 33) else 
    512)
img_exts = ['.jpg', '.png']

jit = "1.7.1" in torch.__version__


class VQGanClip:
    out_dir = ''

    def __init__(
            self, model, perceptor, size=(default_image_size, default_image_size), 
            opt='Adam', step_size=0.1, device=None,
            cut_method='latest', cutn=32, cut_pow=1, augments=None, outputs_dir=None):
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(device)
        self.perceptor = perceptor.to(device)

        self.size = size
        self.sideX, self.sideY, _, _ = size_from_model(size, model)
        emb = model.quantize.embed if is_gumbel(model) else model.quantize.embedding
        self.z_min = emb.weight.min(dim=0).values[None, :, None, None]
        self.z_max = emb.weight.max(dim=0).values[None, :, None, None]
        self.opt_name, self.step_size = opt, step_size
        self.make_cutouts = get_cutout_object(
            cut_method, perceptor.visual.input_resolution, cutn, cut_pow, 
            augments or ['Af', 'Pe', 'Ji', 'Er'])
        self.get_prompts = Promptor(
            perceptor, self.device, self.make_cutouts, (self.sideX, self.sideY))

    def init_image(self, path=None, init_noise=None):
        '''Get the starting image. Either an image from noise (if no image is provided).'''
        if path:
            return read_image(path)
        elif init_noise == 'gradient':
            return random_gradient_image(*self.size)
        return random_noise_image(*self.size)

    def init_images(self, init_image=None, init_noise=None, name=None):
        '''Collect a set of initial images from either a single image, a directory of images, or if neither are provided, from noise.
        '''
        init_images = (
            [f.path for f in os.scandir(init_image) if any(f.path.endswith(e) for e in img_exts) and f.is_file()]
            if os.path.isdir(init_image) else [init_image]) if init_image else []
        return dict([
            (name or os.path.basename(init_image).rsplit('.', 1)[0], self.init_image(init_image, init_noise))
            for init_image in init_images
        ] or [(name or f'noise-{int(time.time())}', self.init_image(None, init_noise))])

    def synth(self, z):
        '''Synthesize from random embedding. embedding -> image'''
        return synth(self.model, z)

    def embed(self, img):
        '''Given a pil image, get its embedding. image -> embedding'''
        img = img.convert('RGB').resize((self.sideX, self.sideY), Image.LANCZOS)
        return self.model.encode(TF.to_tensor(img).to(self.device).unsqueeze(0) * 2 - 1)

    def update(self, z):
        '''Update the optimizer for a new embedding.
        
        Can we do this another way? idk this is what they called everytime the 
        target embeddings (either prompt text or image) were changed.
        '''
        self.opt = get_opt(z, self.opt_name, self.step_size)
        self.z_orig = z_orig = z.clone()
        z.requires_grad_(True)
        # self.z = z
        return z

    def loss(self, z, pMs, i, init_weight=0):
        '''Optimize the model embedding towards a set of prompt embeddings.
        
        Arguments:
            z: The current embedding
            pMs : The prompt embeddings
            i: the step index - used for modifying the init_weight
            initial_weight: how much to weight the initial embedding (i.e. preserving initial image features)
        '''
        self.opt.zero_grad(set_to_none=True)
        im_synth = synth(self.model, z)
        iii = self.perceptor.encode_image(normalize(self.make_cutouts(im_synth))).float()
        
        # gather losses
        losses = [prompt(iii) for prompt in pMs]
        if init_weight:
            losses.append( init_weight * F.mse_loss(z, torch.zeros_like(self.z_orig)) / (2 * (i*2 + 1)) )

        # optimize the loss
        sum(losses).backward()
        self.opt.step()
        with torch.inference_mode():
            z.copy_(z.maximum(self.z_min).minimum(self.z_max))
        return im_synth, losses

    def save_image(self, dir, i, img):
        '''Write the image to file.
        
        Arguments:

        '''
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)
        imageio.imwrite(os.path.join(dir, f'{int(i):03d}.png'), img)

    def run(self, z, pMs, name, n=500, modify_objective=None, video=False, save_every=1, outputs_dir='outputs', debug_freq=10, i=0):
        '''Run VQGAN for a number of steps.
        
        Arguments:
            z: the initial embedding
            pMs: the prompt embeddings
            name (str): the name of the run (used for path names)
            n (int): the number of runs
            modify_objective (callable): a function that has the opportunity modify the objective at each step. E.g. for zoom/rotate
            video (bool): whether you're using this to make a video - if True it will save images along the way (might change this)
            save_every (int): How often to save video frames (in steps)
            outputs_dir (str): The directory to store outputs in. They are stored in {outputs_dir}/{name}/*
            debug_freq (int): How often to write to 'debug.jpg'. Used for showing the current progress of the image.
            i (int): The start iteration. For resuming.


        '''
        self.root_dir = root_dir = os.path.join(outputs_dir, name)
        self.debug_file = debug_file = os.path.join(outputs_dir, name, 'debug.jpg')
        self.frame_dir = steps_dir = os.path.join(outputs_dir, name, 'steps')

        z = self.update(z)
        with tqdm(range(i, i+n)) as pbar:
            for i in pbar:
                if modify_objective:
                    z, pMs = modify_objective(z, pMs, i, root_dir)
                im_synth, losses = self.loss(z, pMs, i)

                if i % debug_freq == 0:
                    checkin(self.model, z, i, losses, debug_file)
                if video and i % save_every == 0:
                    self.save_image(steps_dir, i, im_synth)
        return z

    @classmethod
    def from_config(cls, vqgan_config, vqgan_checkpoint, clip_model='ViT-B/32', **kw):
        '''Helper for loading the model from file.'''
        model = load_vqgan_model(vqgan_config, vqgan_checkpoint)
        perceptor = clip.load(clip_model, jit=jit)[0].eval().requires_grad_(False)
        return cls(model, perceptor, **kw)

    @classmethod
    def load_model(cls, gan_model=None, clip_model=None, vqgan_config=None, vqgan_checkpoint=None, model_dir=model_dir):
        '''Helper for loading the model from file. (I forget the difference).'''
        gan_model = gan_model or 'imagenet_16384'
        return cls(
            load_vqgan_model(
                vqgan_config or os.path.join(model_dir, f'{gan_model}.yml'), 
                vqgan_checkpoint or os.path.join(model_dir, f'{gan_model}.ckpt')), 
            clip.load(clip_model or 'ViT-B/32', jit=jit)[0].eval().requires_grad_(False)
        )


def generate_many(
        prompts=None, image_prompts=(), 
        noise_seeds=(), noise_weights=(), 
        init_image=None, init_noise=None, 
        zoom_start=0, zoom_frequency=10, zoom_dx=0, zoom_dy=0, zoom_scale=1,
        name=None, outputs_dir=None, seed=None, 
        vqgan_config=None, vqgan_checkpoint=None, gan_model=None, clip_model=None,
        **kw):
    '''
    image -> sum([prompt1, prompt2, ...])

    '''
    if seed:
        seed and torch.manual_seed(seed)
        print('Using seed:', torch.seed())

    vq = VQGanClip.load_model(gan_model, clip_model, vqgan_config, vqgan_checkpoint, outputs_dir=outputs_dir, **kw)

    # define prompts

    if not prompts and not image_prompts:
        prompts = "A cute, smiling, Nerdy Rodent"
    # pMs = vq.get_prompts(prompts, images=image_prompts, noise_seeds=noise_seeds, noise_weights=noise_weights)
    pMs_static = vq.get_prompts(images=image_prompts, noise_seeds=noise_seeds, noise_weights=noise_weights)
    pMs_cycle = [vq.get_prompts(p) for p in split_items(prompts, '^')]
    pMs = [*pMs_cycle[:1], *pMs_static]

    # define objective shifting

    def modify_objective(z, pMs, i, out_dir):
        if zoom_scale and zoom_frequency and i % zoom_frequency == 0:
            img = vq.synth(z)
            vq.save_image(os.path.join(out_dir, 'zoom'), i // zoom_frequency, img2uint8(img))
            if zoom_start <= i:
                img = zoom_at(img, vq.sideX/2, vq.sideY/2, zoom_scale, zoom_dx, zoom_dy)
                z = vq.update(vq.embed(img))

        if pMs_cycle and not i % len(pMs_cycle):
            pMs = [pMs_cycle[i//len(pMs_cycle)], *pMs_static]
        return z, pMs

    # get initial images to use
    output_dirs = []
    try:
        # run vq gan and save the output directory for each run
        for im, name_i in vq.init_images(init_image, init_noise, name):
            z, out_dir = vq.run(im, pMs, name_i, modify_objective=modify_objective)
            output_dirs.append(out_dir)
    except KeyboardInterrupt:
        print('Interrupted.')
        if vq.frame_dir and vq.frame_dir not in output_dirs:
            output_dirs.append(vq.frame_dir)
    return output_dirs


def generate_keyframes(vqgan_config, vqgan_checkpoint, clip_model, outputs_dir=None, **kw):
    vq = VQGanClip.from_config(vqgan_config, vqgan_checkpoint, clip_model, outputs_dir=outputs_dir, **kw)
    

