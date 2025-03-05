from tqdm import tqdm
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

ica = np.load("cache/ica.npz")
encoder = ica["encoder"]
decoder = ica["decoder"]
train_mean = ica["mean"]

print('loading data...')
test_images = np.load("data/thingseeg2_metadata/test_images.npy", mmap_mode="r")
test_images = np.array([resize(image, (64, 64)) for image in test_images])
train_images = np.load("data/thingseeg2_metadata/train_images.npy", mmap_mode="r")
train_images = np.array([resize(image, (64, 64)) for image in train_images])

print('extracting features...')

image_dim = 64 * 64 * 3
test_data = np.zeros((len(test_images), image_dim))
for i, image in enumerate(test_images):
    test_data[i, :] = image.flatten()
train_data = np.zeros((len(train_images), image_dim))
for i, image in enumerate(train_images):
    train_data[i, :] = image.flatten()

latent_dim = 1000
test_latents = np.zeros((len(test_data), latent_dim))
for i, image in tqdm(enumerate(test_data)):
    test_latents[i] = encoder[:latent_dim] @ (image - train_mean).reshape(64, 64, 3).T.flatten()
np.save("cache/thingseeg2_extracted_embeddings/test_ica1k.npy", test_latents)

train_latents = np.zeros((len(train_data), latent_dim))
for i, image in tqdm(enumerate(train_data)):
    train_latents[i] = encoder[:latent_dim] @ (image - train_mean).reshape(64, 64, 3).T.flatten()
np.save("cache/thingseeg2_extracted_embeddings/train_ica1k.npy", train_latents)