import setuptools

setuptools.setup(
    name='vqganclip',
    version='0.0.1',
    description='',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    url='https://github.com/nerdyrodent/VQGAN-CLIP',
    packages=setuptools.find_packages(),
    entry_points={'console_scripts': ['vqganclip=generate:main']},
    install_requires=[
        'torchaudio==0.9.0',
        'ftfy', 'regex', 'tqdm', 'omegaconf', 'pytorch-lightning', 'IPython', 'kornia', 'imageio', 'imageio-ffmpeg', 'einops', 'torch_optimizer'
    ],
    extras_require={
        'gpu': [],  # torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
        'cpu': [],  # torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    },
    keywords='vqgan clip image text gan')
