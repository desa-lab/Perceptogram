import numpy as np
import os
from tqdm import tqdm
from PIL import Image

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub = int(args.sub)
param = ''


pred_latents = np.load(f'cache/thingseeg2_preproc/predicted_embeddings/sub-{sub:02d}/regress_pca1k.npy', mmap_mode='r')

recon_dir = f'results/thingseeg2_preproc/sub-{sub:02d}/pca1k{param}/'
os.makedirs(recon_dir, exist_ok=True)

pca = np.load("cache/pca.npz")
eigenvectors = pca["eigenvectors"]
eigenvalues = pca["eigenvalues"]
latent_dim = 1000

print('Reconstructing images...')
images = np.clip(eigenvectors[:latent_dim].T @ pred_latents.T, 0, 1).T.reshape((len(pred_latents), 64, 64, 3), order="F")
images = (images * 255).astype(np.uint8)

print('Saving images...')
for iter in tqdm(range(len(pred_latents)), total=len(pred_latents)):
    img = Image.fromarray(images[iter])
    img.save(f'{recon_dir}{iter:03d}.png')


