import numpy as np
import scipy
from scipy.spatial.distance import correlation
import random
import sklearn.linear_model as skl
import os
import pickle

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument('-weights', '--saving_weights',help="Saving the weights", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('-size', '--size', help='Size', default=8859)
parser.add_argument('-alpha', '--alpha', help='Alpha for regression strength', default=100000)
args = parser.parse_args()
sub = int(args.sub)
saving_weights=args.saving_weights
alpha=int(args.alpha)
param = ''

# Load fMRI data
fmri_train = np.load(f'data/nsd_preproc/sub-{sub:02d}/train_fmriavg_nsdgeneral.npy')
fmri_test = np.load(f'data/nsd_preproc/sub-{sub:02d}/test_fmriavg_nsdgeneral.npy')
fmri_train = fmri_train / 300
fmri_test = fmri_test / 300
norm_mean_train = np.mean(fmri_train, axis=0)
norm_scale_train = np.std(fmri_train, axis=0, ddof=1)
fmri_train = (fmri_train - norm_mean_train) / norm_scale_train
fmri_test = (fmri_test - norm_mean_train) / norm_scale_train
print(fmri_train.shape, fmri_test.shape)

# Save Directory
weights_save_dir = f'cache/nsd_preproc/regression_weights/sub-{sub:02d}/'
os.makedirs(weights_save_dir, exist_ok=True)
weights_filename = f'regress_ica1k_weights{param}.pkl'
save_dir = f'cache/nsd_preproc/predicted_embeddings/sub-{sub:02d}/'
os.makedirs(save_dir, exist_ok=True)
latent_filename = f'regress_ica1k{param}.npy'

# Regression
train_latents= np.load(f'cache/nsd_extracted_embeddings/train_ica1k_sub-{sub:02d}.npy', mmap_mode='r')
test_latents = np.load(f'cache/nsd_extracted_embeddings/test_ica1k.npy', mmap_mode='r')
print(train_latents.shape, test_latents.shape)

print("Training Regression")
reg = skl.Ridge(alpha=alpha, max_iter=50000, fit_intercept=True) # alpha=50000
reg.fit(fmri_train, train_latents)
print('Training complete')

if saving_weights:
    datadict = {
        'weight' : reg.coef_,
        'bias' : reg.intercept_,
    }
    with open(weights_save_dir + weights_filename, "wb") as f:
        pickle.dump(datadict,f)

pred_latent = reg.predict(fmri_test)
pred_latent_mean = np.mean(pred_latent,axis=0)
pred_latent_std = np.std(pred_latent,axis=0)
std_norm_pred_latent = (pred_latent - pred_latent_mean) / pred_latent_std
train_latents_mean = np.mean(train_latents,axis=0)
train_latents_std = np.std(train_latents,axis=0)
pred_latents = std_norm_pred_latent * train_latents_std + train_latents_mean

np.save(save_dir + latent_filename, pred_latents)

# Compute the Euclidean distances
euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_latents, test_latents)])
correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_latents, test_latents)])
# Compute the average Euclidean distance
average_euclidean_distance = euclidean_distances.mean()
correlations = (1 - correlation_distances).mean()
print(reg.score(fmri_test,test_latents), average_euclidean_distance, correlations)

# -0.10514682935676765 41.80231167190395 0.04561628546156888 for 10000
# -0.00597656134904789 41.62013437480925 0.05531842966118934 for 100000
# 0.0010146477466907063 41.731707842050014 0.04857243051499194 for 1000000