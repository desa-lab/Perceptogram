from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd

# image = Image.open('THINGS-images/Images/aardvark/aardvark_01b.jpg')
# image = image.resize((425, 425))

# np_image = np.array(image)
# print(np_image.shape)

# list all items in a folder
import os
# path = 'THINGS-images/Images/aardvark/'
path = 'THINGS-images/Images/'
# items = os.listdir(path)
# print(items)

# list all items in a folder with a specific extension
# import glob
# items = glob.glob(path + '*.jpg')
# print(items)

# list all items in subfolders of a folder, in alphabetical order
import glob
items = sorted(glob.glob(path + '**/*.jpg', recursive=True))
print(len(items))
items1 = [item[21:] for item in items]
print(items1[:20])

# im_dim, im_c = 425, 3
# stim_images = np.zeros((len(items), im_dim, im_dim, im_c))
# print(stim_images.shape)
# for i, item in tqdm.tqdm(enumerate(items)):
#     image = Image.open(item)
#     image = image.resize((im_dim, im_dim))
#     stim_images[i] = np.array(image)

# np.save('data/processed_data/things_stim_images.npy', stim_images)
# print("THINGS images are saved.")

items = pd.read_csv('THINGS-images/Metadata/Image-specific/image_paths.csv', header=None).to_numpy()
items2 = [item[0][7:] for item in items]
print(items.shape)
print(items2[:20])
print(np.setdiff1d(items1, items2))

im_dim, im_c = 425, 3
stim_images = np.zeros((len(items), im_dim, im_dim, im_c))
print(stim_images.shape)
for i, item in tqdm(enumerate(items), total=len(items)):
    item = item[0]
    image = Image.open(path + item[7:])
    image = image.resize((im_dim, im_dim))
    stim_images[i] = np.array(image)

if not os.path.exists('data'):
    os.makedirs('data')
np.save('data/things_stim_images.npy', stim_images)
print("THINGS images are saved.")