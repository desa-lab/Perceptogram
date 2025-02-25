import os
import numpy as np
from PIL import Image
from tqdm import tqdm

img_metadata = np.load('data/thingseeg2_metadata/image_metadata.npy',allow_pickle=True).item()
n_train_img = len(img_metadata['train_img_concepts'])
n_test_img = len(img_metadata['test_img_concepts'])

train_img = np.zeros((n_train_img, 500, 500, 3), dtype=np.uint8)
test_img = np.zeros((n_test_img, 500, 500, 3), dtype=np.uint8)

for train_img_idx in tqdm(range(n_train_img), total=n_train_img, desc='Loading train images'):
    train_img_dir = os.path.join('data/thingseeg2_metadata', 'training_images',
        img_metadata['train_img_concepts'][train_img_idx],
        img_metadata['train_img_files'][train_img_idx])
    train_img[train_img_idx] = np.array(Image.open(train_img_dir).convert('RGB'))

np.save('data/thingseeg2_metadata/train_images.npy', train_img)

for test_img_idx in tqdm(range(n_test_img), total=n_test_img, desc='Loading test images'):
    test_img_dir = os.path.join('data/thingseeg2_metadata', 'test_images',
        img_metadata['test_img_concepts'][test_img_idx],
        img_metadata['test_img_files'][test_img_idx])
    test_img[test_img_idx] = np.array(Image.open(test_img_dir).convert('RGB'))

np.save('data/thingseeg2_metadata/test_images.npy', test_img)

test_images_dir = 'data/thingseeg2_metadata/test_images_direct/'

if not os.path.exists(test_images_dir):
   os.makedirs(test_images_dir)
for i in tqdm(range(len(test_img)), total=len(test_img), desc='Saving direct test images'):
    im = Image.fromarray(test_img[i].astype(np.uint8))
    im.save('{}/{}.png'.format(test_images_dir,i))