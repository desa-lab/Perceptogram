import numpy as np
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub = int(args.sub)

images = np.load(f'data/nsd_metadata/test_images.npy', mmap_mode='r')
test_images_dir = f'data/nsd_metadata/test_images_direct/'

if not os.path.exists(test_images_dir):
   os.makedirs(test_images_dir)
for i in range(len(images)):
    im = Image.fromarray(images[i].astype(np.uint8))
    im.save(os.path.join(test_images_dir, f"{i}.png"))


