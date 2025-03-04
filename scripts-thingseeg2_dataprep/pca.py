import numpy as np
import time, os
from tqdm import tqdm

# constants
num_train_samples = 1281167
num_val_samples = 50000
num_test_samples = 16740
image_dim = 64 * 64 * 3

# reshape and rotate 90 degrees counter-clockwise
def format(image):
    return np.rot90(np.reshape(image, (64, 64, 3), order="F"), k=-1)

# load images
start_time = time.time()  # Start timing
train_data = np.zeros((num_train_samples, image_dim))
for i in tqdm(range(9)):
    batch = np.load(f"data/imagenet64/train_data_batch_{i + 1}.npz")
    # print(batch["data"].shape)
    train_data[i * 128116:(i + 1) * 128116, :] = batch["data"] / 255
batch = np.load("data/imagenet64/train_data_batch_10.npz")
train_data[9 * 128116:, :] = batch["data"] / 255
val_data = np.load("data/imagenet64/val_data.npz")["data"] / 255
end_time = time.time()  # End timing
print(f"Time taken to load images: {end_time - start_time} seconds") # 352 seconds

start_time = time.time()  # Start timing
train_mean = np.mean(train_data, axis=0)
np.save("data/imagenet64/train_mean.npy", train_mean)
end_time = time.time()  # End timing
print(f"Time taken to compute mean: {end_time - start_time} seconds") # 45 seconds

# subtract mean and save
start_time = time.time()  # Start timing
train_mean = np.load("data/imagenet64/train_mean.npy")
for i in tqdm(range(10)):
    batch = np.load(f"data/imagenet64/train_data_batch_{i + 1}.npz")
    # pre-whiten to save memory
    mean_subtracted_batch = (batch["data"] / 255 - train_mean)
    np.save(f"data/imagenet64/train_data_batch_mean_subtracted_{i + 1}.npy", mean_subtracted_batch)
end_time = time.time()  # End timing
print(f"Time taken to subtract mean: {end_time - start_time} seconds") # 3304 seconds

# load mean-subtracted images
start_time = time.time()  # Start timing
train_data = np.zeros((num_train_samples, image_dim))
for i in tqdm(range(9)):
    train_data[i * 128116:(i + 1) * 128116, :] = np.load(f"data/imagenet64/train_data_batch_mean_subtracted_{i + 1}.npy")
train_data[9 * 128116:, :] = np.load(f"data/imagenet64/train_data_batch_mean_subtracted_{10}.npy")
end_time = time.time()  # End timing
print(f"Time taken to load mean-subtracted images: {end_time - start_time} seconds") # 163 seconds

start_time = time.time()  # Start timing
cov = np.cov(train_data, rowvar=False)
os.makedirs('cache/imagenet64', exist_ok=True)
np.save("cache/imagenet64/cov.npy", cov)
end_time = time.time()  # End timing
print(f"Time taken to compute covariance: {end_time - start_time} seconds") # 718 seconds
# cov = np.load("cache/imagenet64/cov.npy")


# compute PCA
start_time = time.time()  # Start timing
eigenvalues, eigenvectors = np.linalg.eigh(cov)
eigenvectors = np.fliplr(eigenvectors).T
eigenvalues = np.flip(eigenvalues)
np.savez('cache/pca.npz', eigenvectors=eigenvectors, eigenvalues=eigenvalues)
end_time = time.time()  # End timing
print(f"Time taken to compute PCA: {end_time - start_time} seconds") # 117 seconds