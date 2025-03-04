import numpy as np
import os
from tqdm import tqdm
from PIL import Image

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument('-size', '--size', help='Size', default=8859)
args = parser.parse_args()
sub = int(args.sub)
param = ''


pred_latents = np.load(f'cache/nsd_preproc/predicted_embeddings/sub-{sub:02d}/regress_ica1k.npy', mmap_mode='r')

recon_dir = f'results/nsd_preproc/sub-{sub:02d}/ica1k{param}/'
os.makedirs(recon_dir, exist_ok=True)

ica = np.load("cache/ica.npz")
encoder = ica["encoder"]
decoder = ica["decoder"]
train_mean = ica["mean"]
latent_dim = 1000

print('Reconstructing images...')
images = np.clip(decoder[:, :latent_dim] @ pred_latents.T + train_mean[:, np.newaxis], 0, 1).T.reshape((len(pred_latents), 64, 64, 3), order="F")
images = (images * 255).astype(np.uint8)

print('Saving images...')
for iter in tqdm(range(len(pred_latents)), total=len(pred_latents)):
    img = Image.fromarray(images[iter])
    img.save(f'{recon_dir}{iter:03d}.png')


