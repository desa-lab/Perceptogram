import numpy as np
import pickle
import matplotlib.pyplot as plt
from nilearn.image import load_img, index_img
import nibabel as nib
import cortex
import os

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub = int(args.sub)

# Load data
with open(f'cache/nsd_preproc/regression_weights/sub-{sub:02d}/regress-encode_clip_weights.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight'].T
    reg_b = datadict['bias']
print(reg_w.shape, reg_b.shape)

train_latents= np.load(f'cache/nsd_extracted_embeddings/train_clip_sub-{sub:02d}.npy')
test_latents = np.load(f'cache/nsd_extracted_embeddings/test_clip.npy')
print(train_latents.shape, test_latents.shape)

train_latents_mean = np.mean(train_latents,axis=0)
train_latents_std = np.std(train_latents,axis=0)
train_latents_whitened = (train_latents - train_latents_mean) / train_latents_std
test_latents_whitened = (test_latents - train_latents_mean) / train_latents_std
new_w = test_latents @ reg_w
new_w_whitened = test_latents_whitened @ reg_w

cov_train_latents = np.cov(train_latents, rowvar=False)
cov_train_latents_whitened = np.cov(train_latents_whitened, rowvar=False)
print(cov_train_latents_whitened.shape, cov_train_latents.shape)

fmri_train = np.load(f'data/nsd_preproc/sub-{sub:02d}/train_fmriavg_nsdgeneral.npy')
fmri_test = np.load(f'data/nsd_preproc/sub-{sub:02d}/test_fmriavg_nsdgeneral.npy')
fmri_train = fmri_train.reshape(fmri_train.shape[0],-1)
fmri_test = fmri_test.reshape(fmri_test.shape[0],-1)
print(fmri_train.shape, fmri_test.shape)
norm_mean_train = np.mean(fmri_train, axis=0)
norm_scale_train = np.std(fmri_train, axis=0, ddof=1)
fmri_train_whitened = (fmri_train - norm_mean_train) / norm_scale_train
fmri_test_whitened = (fmri_test - norm_mean_train) / norm_scale_train

pred_latents = np.load(f'cache/nsd_preproc/predicted_embeddings/sub-{sub:02d}/regress_clip.npy')
pred_latents_whitened = (pred_latents - train_latents_mean) / train_latents_std
new_w_pred = pred_latents @ reg_w
new_w_pred_whitened = pred_latents_whitened @ reg_w
print(new_w_pred.shape, new_w_pred_whitened.shape)

# Reorder the test images
from scipy.spatial.distance import pdist
dist_test_latents_whitened = pdist(test_latents_whitened, metric='correlation') # braycurtis with average
from scipy.cluster.hierarchy import linkage, leaves_list
linkage_data = linkage(dist_test_latents_whitened, method='average', metric='correlation')
print(linkage_data.shape)
leaves = leaves_list(linkage_data)

## (See scripts-nsd/visualize_unCLIP_patterns.ipynb for the derivation of the clusters)
map1 = new_w_pred_whitened[leaves][:223].mean(axis=0) # animals in nature
map2 = new_w_pred_whitened[leaves][223:223+140].mean(axis=0) # room interiors
map3 = new_w_pred_whitened[leaves][223+140:223+140+62].mean(axis=0) # human closeup
map4 = new_w_pred_whitened[leaves][223+140+62:223+140+62+88].mean(axis=0) # food (some miscellaneous)
map5 = new_w_pred_whitened[leaves][223+140+62+88:223+140+62+88+222].mean(axis=0) # human from a distance
map6 = new_w_pred_whitened[leaves][223+140+62+88+222:].mean(axis=0) # urban scenes

# Get a random fMRI volumn and substitute its voxels with the patterns
betas = load_img(f'data/nsd_preproc/subj{sub:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz')
betas_trial = index_img(betas, 0)
roi_dir = f'data/nsd_preproc/subj{sub:02d}/func1pt8mm/roi/'
mask_filename = 'nsdgeneral.nii.gz'
mask = nib.load(roi_dir+mask_filename).get_fdata()
betas_trial_masked = betas_trial.get_fdata()

os.makedirs(f'results/nsd_preproc/sub-{sub:02d}/clip_patterns/func1pt8mm/', exist_ok=True)
os.makedirs(f'cache/nsd_preproc/predicted_patterns/clip_patterns/sub-{sub:02d}/func1pt8mm/', exist_ok=True)
maps = [map1, map2, map3, map4, map5, map6]
pattern_names = ['animals', 'interiors', 'human-closeup', 'food', 'human-distant', 'urban']
for i_map, map in enumerate(maps):
    # plot subject-native pattern
    betas_trial_masked[mask<1] = 0
    betas_trial_masked[mask==1] = map
    vol_data = cortex.Volume(np.moveaxis(np.moveaxis(betas_trial.get_fdata(),0,-1),0,1), f'subj{sub:02d}', 'full', cmap='twilight', vmin=-1.5, vmax=1.5)
    fig = plt.figure(dpi=50) # 100
    cortex.quickflat.make_figure(vol_data, recache=1, fig=fig)
    plt.title(f'{pattern_names[i_map].capitalize()} Pattern - NSD Subject {sub} 1.8mm')
    plt.savefig(f'results/nsd_preproc/sub-{sub:02d}/clip_patterns/func1pt8mm/{pattern_names[i_map]}_pattern.png', dpi=300)    
    plt.close()

    pattern_image = nib.Nifti1Image(betas_trial_masked, affine=betas_trial.affine, header=betas_trial.header)
    nib.save(pattern_image, f'cache/nsd_preproc/predicted_patterns/clip_patterns/sub-{sub:02d}/func1pt8mm/{pattern_names[i_map]}_pattern.nii.gz')