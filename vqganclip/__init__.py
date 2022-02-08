import os
from .vqgan import *
opj = os.path.join

def video_styler(video_file, prompt, output_dir=None, seed=12345):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    frames_dir = video2frames(video_file)

    output_dir = output_dir or f'{frames_dir}-styled'
    frame_files = []
    for f in glob.glob(os.path.join(frames_dir, '*')):
        f_out = generate_many(prompts=prompt, init_image=f, output_dir=output_dir, opt='Adagrad', lr=0.2, n=25, size=(640, 360), seed=seed, debug=True)
        frame_files.append(f_out)
    frames2video(frame_files)


def so_random():
    import random
    text_one=("A painting of a", "A pencil art sketch of a", "An illustration of a", "A photograph of a")
    text_two=("spinning", "dreaming", "watering", "loving", "eating", "drinking", "sleeping", "repeating", "surreal", "psychedelic")
    text_three=(
        "fish", "egg", "peacock", "watermelon", "pickle", "horse", "dog", "house", "kitchen", "bedroom", "door", "table", "lamp", "dresser", "watch", "logo", "icon", "tree",
        "grass", "flower", "plant", "shrub", "bloom", "screwdriver", "spanner", "figurine", "statue", "graveyard", "hotel", "bus", "train", "car", "lamp", "computer", "monitor")
    styles=(
        "Art Nouveau", "Camille Pissarro", "Michelan,gelo Caravaggio", "Claude Monet", "Edgar Degas", "Edvard Munch", "Fauvism", "Futurism", "Impressionism",
        "Picasso", "Pop Art", "Modern art", "Surreal Art", "Sandro Botticelli", "oil paints", "watercolours", "weird bananas", "strange colours")

    
    for i in range(50):
        text = ' '.join([
            random.choice(text_one),
            random.choice(text_two),
            random.choice(text_three), 
            'and a', 
            random.choice(text_three),
            'in the style of', 
            random.choice(styles), 
            'and', 
            random.choice(styles),
        ])
        generate_many(text, name=f'random-{i}')


def zoom(text, file, max_epochs=180, output_dir=None, lr=0.1, opt='Adam', max_iterations=25, seed=12345):
    import shutil
    output_dir = output_dir or f'{text}-zoomed'
    frames = []
    for i in range(max_epochs):
        generate_many(prompt=text, output_dir=output_dir, opt=opt, lr=lr, max_iterations=max_iterations, save_every=max_iterations, seed=seed)
        fname2 = f'{0}{2:04d}{1}'.format(*os.path.splitext(file), i)
        frames.append(fname2)
        # cp file fname2
        # convert "$FILENAME" -distort SRT 1.01,0 -gravity center "$FILENAME"	# Zoom
        # convert "$FILENAME" -distort SRT 1 -gravity center "$FILENAME"	# Rotate
    video2frames(frames)


def opt_tester(text="A painting in the style of Paul Gauguin", output_dir='OptimiserTesting-60it-Noise-NPW-1', iterations=60, size=(256, 256), seed=12345):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    opts = [
        ("Adam", .1, .1, 1)
        ("AdamW", .1, .1, 1)
        ("Adamax", .1, .1, 1)
        ("Adagrad", .1, .25, 1)
        ("AdamP", .1, .25, 1)
        ("RAdam", .1, .25, 1)
        ("DiffGrad", .1, .25, 1)
    ]
    for opt, lr, step, npw in opts:
        for i in range(10):
            generate_many(
                prompt=text, init_noise='pixels', init_weight=1, noise_seeds=666, noise_weights=npw, 
                output_file=opj(output_dir, f'{i:04d}-{opt}-{lr}-{npw}.png'), # "$OUT_DIR"/"$PADDED_COUNT"-"$OPTIMISER"-"$LR"-"$NPW".png,
                opt=opt, lr=lr, iterations=iterations, save_every=iterations, 
                size=size, seed=seed, debug=True)
            lr = lr + step

    #!!!!!!!!!
    # # Make montage
    # mogrify -font Liberation-Sans -fill white -undercolor '#00000080' -pointsize 14 -gravity NorthEast -annotate +10+10 %t "$OUT_DIR"/*.png
    # montage "$OUT_DIR"/*.png -geometry 256x256+1+1 -tile 10x7 collage.jpg
    