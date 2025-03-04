from sklearn.decomposition import FastICA
import numpy as np
import time, os
from tqdm import tqdm

pca = np.load("cache/pca.npz")
eigenvectors = pca["eigenvectors"].T
eigenvalues = pca["eigenvalues"]
D = np.diag(1.0 / np.sqrt(eigenvalues[:1000]))
whitening_matrix = D @ eigenvectors[:, :1000].T

# constants
num_train_samples = 1281167
num_val_samples = 50000
num_test_samples = 16740
image_dim = 64 * 64 * 3
num_components = 1000

# reshape and rotate 90 degrees counter-clockwise
def format(image):
    return np.rot90(np.reshape(image, (64, 64, 3), order="F"), k=-1)

# load images
start_time = time.time()  # Start timing
train_data = np.zeros((num_train_samples, image_dim))
for i in tqdm(range(9)):
    batch = np.load(f"data/imagenet64/train_data_batch_{i + 1}.npz")
    train_data[i * 128116:(i + 1) * 128116, :] = batch["data"] / 255
batch = np.load("data/imagenet64/train_data_batch_10.npz")
train_data[9 * 128116:, :] = batch["data"] / 255
end_time = time.time()  # End timing
print(f"Time taken to load images: {end_time - start_time} seconds") # 190 seconds

train_mean = np.mean(train_data, axis=0)
np.save("data/imagenet64/train_mean.npy", train_mean)

train_mean = np.load("data/imagenet64/train_mean.npy")
# load images
start_time = time.time()  # Start timing
for i in tqdm(range(9)):
    batch = np.load(f"data/imagenet64/train_data_batch_{i + 1}.npz")
    # pre-whiten to save memory
    whitened_batch = (batch["data"] / 255 - train_mean) @ whitening_matrix.T
    np.save(f"data/imagenet64/whitened_batch_{i + 1}.npy", whitened_batch)
batch = np.load("data/imagenet64/train_data_batch_10.npz")
whitened_batch = (batch["data"] / 255 - train_mean) @ whitening_matrix.T
np.save(f"data/imagenet64/whitened_batch_{10}.npy", whitened_batch)
end_time = time.time()  # End timing
print(f"Time taken to whiten images: {end_time - start_time} seconds") # 3304 seconds

# loading whitened data
start_time = time.time()  # Start timing
whitened_data = np.zeros((num_train_samples, num_components), dtype=np.float32) # saving memory by not using float64
for i in range(9):
    print(i)
    whitened_data[i * 128116:(i + 1) * 128116, :] = np.load(f"data/imagenet64/whitened_batch_{i + 1}.npy")
whitened_data[9 * 128116:, :] = np.load(f"data/imagenet64/whitened_batch_10.npy")
end_time = time.time()  # End timing
print(f"Time taken to load whitened images: {end_time - start_time} seconds") # 38 seconds

start_time = time.time()  # Start timing
ica1000 = FastICA(random_state=0, whiten=False, tol=0.001)
# ica = FastICA(n_components=1000, random_state=0, whiten=False, tol=0.001)
ica1000.fit(whitened_data)
np.save("cache/ica1k_components.npy", ica1000.components_)
end_time = time.time()  # End timing
print(f"Time taken to compute ICA: {end_time - start_time} seconds") # 3707 seconds

start_time = time.time()  # Start timing
components_1000 = np.load("cache/ica1k_components.npy")
encoder = components_1000 @ whitening_matrix
decoder = np.linalg.pinv(encoder)
np.savez("cache/ica.npz", encoder=encoder, decoder=decoder, mean=train_mean)
end_time = time.time()  # End timing
print(f"Time taken to save encoder and decoder: {end_time - start_time} seconds") # 7 seconds