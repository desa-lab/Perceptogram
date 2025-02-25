import numpy as np
import os
from PIL import Image

#The same for all subjects
# images = np.load('data/processed_data/subj01/nsd_test_stim_sub1.npy')

# images = np.load('cache/processed_data/BIGMEG1/test_images1b_sub-BIGMEG1.npy', mmap_mode='r')
# test_images_dir = 'cache/thingsmeg_stimuli/test_images1b/'

images = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_images1b_sub-BIGMEG1.npy', mmap_mode='r')
test_images_dir = 'cache/thingsmeg_stimuli/avg_test_images1b/'

if not os.path.exists(test_images_dir):
   os.makedirs(test_images_dir)
for i in range(len(images)):
    im = Image.fromarray(images[i].astype(np.uint8))
    im.save('{}/{}.png'.format(test_images_dir,i))


