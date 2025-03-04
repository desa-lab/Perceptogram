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
norm_mean_test = np.mean(fmri_test, axis=0)
norm_scale_test = np.std(fmri_test, axis=0, ddof=1)
fmri_train = (fmri_train - norm_mean_train) / norm_scale_train
fmri_test = (fmri_test - norm_mean_test) / norm_scale_test
print(fmri_train.shape, fmri_test.shape)

# Save Directory
weights_save_dir = f'cache/nsd_preproc/regression_weights/sub-{sub:02d}/'
os.makedirs(weights_save_dir, exist_ok=True)
weights_filename = f'regress-encode_ica1k_weights{param}.pkl'
save_dir = f'cache/nsd_preproc/predicted_fmri/sub-{sub:02d}/'
os.makedirs(save_dir, exist_ok=True)
pred_filename = f'regress-encode_ica1k{param}.npy'
 
# Regression
train_latents= np.load(f'cache/nsd_extracted_embeddings/train_ica1k_sub-{sub:02d}.npy', mmap_mode='r')
test_latents = np.load(f'cache/nsd_extracted_embeddings/test_ica1k.npy', mmap_mode='r')
print(train_latents.shape, test_latents.shape)
train_latents_mean = np.mean(train_latents,axis=0)
train_latents_std = np.std(train_latents,axis=0)
train_latents = (train_latents - train_latents_mean) / train_latents_std
test_latents = (test_latents - train_latents_mean) / train_latents_std

print("Training Regression")
reg = skl.Ridge(alpha=alpha, max_iter=50000, fit_intercept=True) # alpha=50000
reg.fit(train_latents, fmri_train)
print('Training complete')

if saving_weights:
    datadict = {
        'weight' : reg.coef_,
        'bias' : reg.intercept_,
    }
    with open(weights_save_dir + weights_filename, "wb") as f:
        pickle.dump(datadict,f)

pred_fmri = reg.predict(test_latents)

np.save(save_dir + pred_filename, pred_fmri)

# Compute the Euclidean distances
euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_fmri, fmri_test)])
correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_fmri, fmri_test)])
# Compute the average Euclidean distance
average_euclidean_distance = euclidean_distances.mean()
correlations = (1 - correlation_distances).mean()
print(reg.score(test_latents,fmri_test), average_euclidean_distance, correlations)

# -0.049982203320936724 127.20013624453078 0.09714771144121306 for 1000
# 0.005101180219735448 123.7703657239564 0.10355592552118774 for 10000
# 0.0059998074077347975 123.70748514248484 0.1086088811945179 for 100000
# 0.0007953853164131198 124.03513047615974 0.10938870137874411 for 1000000
