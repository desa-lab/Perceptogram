import numpy as np
import pickle
import matplotlib.pyplot as plt
from nilearn.image import load_img, index_img
import nibabel as nib
import cortex
import os
import cv2

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub = int(args.sub)

# Load data
with open(f'cache/nsd_preproc/regression_weights/sub-{sub:02d}/regress-encode_pca1k_weights.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight'].T
    reg_b = datadict['bias']
print(reg_w.shape, reg_b.shape)

train_latents= np.load(f'cache/nsd_extracted_embeddings/train_pca1k_sub-{sub:02d}.npy')
test_latents = np.load(f'cache/nsd_extracted_embeddings/test_pca1k.npy')
test_text = np.load(f'data/nsd_metadata/test_texts.npy')
test_images = np.load(f'data/nsd_metadata/test_images.npy')
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

pred_latents = np.load(f'cache/nsd_preproc/predicted_embeddings/sub-{sub:02d}/regress_pca1k.npy')
pred_latents_whitened = (pred_latents - train_latents_mean) / train_latents_std
new_w_pred = pred_latents @ reg_w
new_w_pred_whitened = pred_latents_whitened @ reg_w
print(new_w_pred.shape, new_w_pred_whitened.shape)

# Reorder the test images
def preprocess_image(image, downsize_factor=1, blur_ksize=(7, 7)):
    # Downsize the image
    height, width = image.shape[:2]
    new_size = (int(width * downsize_factor), int(height * downsize_factor))
    downsized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    # Blur the image
    blurred_image = cv2.GaussianBlur(downsized_image, blur_ksize, 0)
    
    return blurred_image

def calculate_average_color(image):
    # Calculate the average color of the image
    return np.mean(image, axis=(0, 1))

def calculate_luminance(color):
    # Calculate the luminance of the color using the formula for perceived luminance
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]

def calculate_hue(color):
    # Convert RGB to HSV and return the hue
    color = np.uint8([[color]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    hue = hsv_color[0][0][0]
    # Adjust hue to handle wrap-around for red colors
    if hue > 160:
        hue -= 180
    return hue

pca = np.load("cache/pca.npz")
eigenvectors = pca["eigenvectors"]
eigenvalues = pca["eigenvalues"]
latent_dim = 1000
pred_images = np.clip(eigenvectors[:latent_dim].T @ pred_latents.T, 0, 1).T.reshape((len(pred_latents), 64, 64, 3), order="F")
pred_images = (pred_images * 255).astype(np.uint8)

# Preprocess images and calculate the average color for each image
preprocessed_images = [preprocess_image(image) for image in pred_images]
average_colors = [calculate_average_color(image) for image in preprocessed_images]

# Calculate luminance and hue for each average color
luminances = [calculate_luminance(color) for color in average_colors]
hues = [calculate_hue(color) for color in average_colors]

# Get the argsort indexes for luminance and hue
luminance_argsort = np.argsort(luminances)
hue_argsort = np.argsort(hues)

# map1 = new_w_pred_whitened[luminance_argsort][0:200].mean(axis=0) # Dark
# map2 = new_w_pred_whitened[luminance_argsort][-200:].mean(axis=0) # Bright
map1 = new_w_pred[luminance_argsort][0:200].mean(axis=0) # Dark
map2 = new_w_pred[luminance_argsort][-200:].mean(axis=0) # Bright

# Get a random fMRI volumn and substitute its voxels with the patterns
betas = load_img(f'data/nsd_preproc/subj{sub:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz')
betas_trial = index_img(betas, 0)
roi_dir = f'data/nsd_preproc/subj{sub:02d}/func1pt8mm/roi/'
mask_filename = 'nsdgeneral.nii.gz'
mask = nib.load(roi_dir+mask_filename).get_fdata()
betas_trial_masked = betas_trial.get_fdata()

os.makedirs(f'results/nsd_preproc/sub-{sub:02d}/pca-brightness_patterns/func1pt8mm/', exist_ok=True)
os.makedirs(f'cache/nsd_preproc/predicted_patterns/pca-brightness_patterns/sub-{sub:02d}/func1pt8mm/', exist_ok=True)
maps = [map1, map2]
pattern_names = ['dark', 'bright']
for i_map, map in enumerate(maps):
    # plot subject-native pattern
    betas_trial_masked[mask<1] = 0
    betas_trial_masked[mask==1] = map
    vol_data = cortex.Volume(np.moveaxis(np.moveaxis(betas_trial.get_fdata(),0,-1),0,1), f'subj{sub:02d}', 'full', cmap='twilight', vmin=-1., vmax=1.)
    fig = plt.figure(dpi=50) # 100
    cortex.quickflat.make_figure(vol_data, recache=1, fig=fig)
    plt.title(f'{pattern_names[i_map].capitalize()} Pattern - NSD Subject {sub} 1.8mm')
    plt.savefig(f'results/nsd_preproc/sub-{sub:02d}/pca-brightness_patterns/func1pt8mm/{pattern_names[i_map]}_pattern.png', dpi=300)    
    plt.close()

    pattern_image = nib.Nifti1Image(betas_trial_masked, affine=betas_trial.affine, header=betas_trial.header)
    nib.save(pattern_image, f'cache/nsd_preproc/predicted_patterns/pca-brightness_patterns/sub-{sub:02d}/func1pt8mm/{pattern_names[i_map]}_pattern.nii.gz')