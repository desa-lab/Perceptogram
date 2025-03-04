# %%
import numpy as np
import pickle
import sklearn.linear_model as skl
import os
from scipy.spatial.distance import correlation
from tqdm import tqdm

# %% [markdown]
### Predict VDVAE Features

# %%
# with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_autokl1b_weights_sub-BIGMEG1.pkl',"rb") as f:
# with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_autokl1b_weights_sub-BIGMEG1_200ms.pkl',"rb") as f:
# with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_autokl1b_weights_sub-BIGMEG1_400ms.pkl',"rb") as f:
# with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_autokl1b_weights_sub-BIGMEG1_600ms.pkl',"rb") as f:
with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_autokl1b_weights_sub-BIGMEG1_800ms.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']

# %%
reg = skl.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)
reg.coef_ = reg_w
reg.intercept_ = reg_b

# %%
# averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1.npy')
# averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1_200ms.npy')
# averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1_400ms.npy')
# averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1_600ms.npy')
averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1_800ms.npy')
averaged_epochs = averaged_epochs.reshape(averaged_epochs.shape[0], -1) # concatenate all time points
train_latents = np.load('cache/thingsmeg/extracted_embeddings/BIGMEG1/train_autokl1b_sub-BIGMEG1.npy', mmap_mode='r')

# %%
pred_test_latent = reg.predict(averaged_epochs)
std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
pred_latents = std_norm_test_latent * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)

# %%
subject = 'BIGMEG1'
save_dir = 'cache/thingsmeg/predicted_embeddings/' + subject + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# np.save(save_dir + f'avg_thingsmeg_regress_autokl1b_sub-{subject}.npy', pred_latents)
# np.save(save_dir + f'avg_thingsmeg_regress_autokl1b_sub-{subject}_200ms.npy', pred_latents)
# np.save(save_dir + f'avg_thingsmeg_regress_autokl1b_sub-{subject}_400ms.npy', pred_latents)
# np.save(save_dir + f'avg_thingsmeg_regress_autokl1b_sub-{subject}_600ms.npy', pred_latents)
np.save(save_dir + f'avg_thingsmeg_regress_autokl1b_sub-{subject}_800ms.npy', pred_latents)

# %% [markdown]
# Please run `vdvae_reconstruct_images1b.py` or `vdvae_reconstruct_images.py` after this

# %% [markdown]
# ### Predict CLIP-Text Features

# %%
# with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_cliptext1bcategoryalltokens_weights_sub-BIGMEG1.pkl',"rb") as f:
# with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_cliptext1bcategoryalltokens_weights_sub-BIGMEG1_200ms.pkl',"rb") as f:
# with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_cliptext1bcategoryalltokens_weights_sub-BIGMEG1_400ms.pkl',"rb") as f:
# with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_cliptext1bcategoryalltokens_weights_sub-BIGMEG1_600ms.pkl',"rb") as f:
with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_cliptext1bcategoryalltokens_weights_sub-BIGMEG1_800ms.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']

# %%
# averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1.npy')
# averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1_200ms.npy')
# averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1_400ms.npy')
# averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1_600ms.npy')
averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1_800ms.npy')
averaged_epochs = averaged_epochs.reshape(averaged_epochs.shape[0], -1) # concatenate all time points
train_clip = np.load('cache/thingsmeg/extracted_embeddings/BIGMEG1/train_cliptext1bcategory_sub-BIGMEG1.npy', mmap_mode='r')
# test_clip = np.load('cache/extracted_embeddings/BIGMEG1/test_cliptext1bcategory_sub-BIGMEG1.npy', mmap_mode='r')

# %%
num_embed = 77
pred_clip = np.zeros((averaged_epochs.shape[0], num_embed, 768))
for i in tqdm(range(num_embed), total=num_embed):
    reg = skl.Ridge(alpha=100000, max_iter=50000, fit_intercept=True) # old alpha=100000, optimal alpha=1200000

    reg.coef_ = reg_w[i]
    reg.intercept_ = reg_b[i]
    
    pred_test_latent = reg.predict(averaged_epochs)
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
    pred_clip[:,i] = std_norm_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)

    # # Compute the Euclidean distances
    # euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    # correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    # # Compute the average Euclidean distance
    # average_euclidean_distance = euclidean_distances.mean()
    # correlations = (1 - correlation_distances).mean()

    # print(i,reg.score(averaged_epochs,test_clip[:,i]), average_euclidean_distance, correlations)

# %%
subject = 'BIGMEG1'
save_dir = 'cache/thingsmeg/predicted_embeddings/' + subject + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# np.save(save_dir + f'avg_thingsmeg_regress_cliptext1bcategoryalltokens_sub-{subject}.npy', pred_clip)
# np.save(save_dir + f'avg_thingsmeg_regress_cliptext1bcategoryalltokens_sub-{subject}_200ms.npy', pred_clip)
# np.save(save_dir + f'avg_thingsmeg_regress_cliptext1bcategoryalltokens_sub-{subject}_400ms.npy', pred_clip)
# np.save(save_dir + f'avg_thingsmeg_regress_cliptext1bcategoryalltokens_sub-{subject}_600ms.npy', pred_clip)
np.save(save_dir + f'avg_thingsmeg_regress_cliptext1bcategoryalltokens_sub-{subject}_800ms.npy', pred_clip)

# %% [markdown]
# ### Predict CLIP-Vision Features

# %%
# with open('cache/regression_weights/BIGMEG1/thingsmeg_regress_clipvision1b_weights_sub-BIGMEG1_precomputed.pkl',"rb") as f:
# with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_clipvision1b_weights_sub-BIGMEG1.pkl',"rb") as f:
# with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_clipvision1b_weights_sub-BIGMEG1_200ms.pkl',"rb") as f:
# with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_clipvision1b_weights_sub-BIGMEG1_400ms.pkl',"rb") as f:
# with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_clipvision1b_weights_sub-BIGMEG1_600ms.pkl',"rb") as f:
with open('cache/thingsmeg/regression_weights/BIGMEG1/thingsmeg_regress_clipvision1b_weights_sub-BIGMEG1_800ms.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']

# %%
# averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1.npy')
# averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1_200ms.npy')
# averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1_400ms.npy')
# averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1_600ms.npy')
averaged_epochs = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_avg_thingsmeg_sub-BIGMEG1_800ms.npy')
averaged_epochs = averaged_epochs.reshape(averaged_epochs.shape[0], -1) # concatenate all time points
train_clip = np.load('cache/thingsmeg/extracted_embeddings/BIGMEG1/train_clipvision1b_sub-BIGMEG1.npy', mmap_mode='r')

# %%
num_embed = 257
pred_clip = np.zeros((averaged_epochs.shape[0], num_embed, 768))
for i in tqdm(range(num_embed), total=num_embed):
    reg = skl.Ridge(alpha=60000, max_iter=50000, fit_intercept=True) # old alpha=60000, optimal alpha=300000
    
    reg.coef_ = reg_w[i]
    reg.intercept_ = reg_b[i]
    
    pred_test_latent = reg.predict(averaged_epochs)
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
    pred_clip[:,i] = std_norm_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)

    # # Compute the Euclidean distances
    # euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    # correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    # # Compute the average Euclidean distance
    # average_euclidean_distance = euclidean_distances.mean()
    # correlations = (1 - correlation_distances).mean()
    
    # print(i,reg.score(test_fmri,test_clip[:,i]), average_euclidean_distance, correlations)

# %%
subject = 'BIGMEG1'
save_dir = 'cache/thingsmeg/predicted_embeddings/' + subject + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# np.save(save_dir + f'avg_thingsmeg_regress_clipvision1b_sub-{subject}_precomputed.npy', pred_clip)
# np.save(save_dir + f'avg_thingsmeg_regress_clipvision1b_sub-{subject}.npy', pred_clip)
# np.save(save_dir + f'avg_thingsmeg_regress_clipvision1b_sub-{subject}_200ms.npy', pred_clip)
# np.save(save_dir + f'avg_thingsmeg_regress_clipvision1b_sub-{subject}_400ms.npy', pred_clip)
# np.save(save_dir + f'avg_thingsmeg_regress_clipvision1b_sub-{subject}_600ms.npy', pred_clip)
np.save(save_dir + f'avg_thingsmeg_regress_clipvision1b_sub-{subject}_800ms.npy', pred_clip)

# %%



