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
with open(f'cache/nsd_preproc/regression_weights/sub-{sub:02d}/regress-encode_vdvae_weights.pkl',"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight'].T
    reg_b = datadict['bias']
print(reg_w.shape, reg_b.shape)

train_latents= np.load(f'cache/nsd_extracted_embeddings/train_vdvae_sub-{sub:02d}.npy')
test_latents = np.load(f'cache/nsd_extracted_embeddings/test_vdvae.npy')
test_text = np.load(f'data/nsd_metadata/test_texts.npy')
test_images = np.load(f'data/nsd_metadata/test_images.npy')
print(train_latents.shape, test_latents.shape)

train_latents_mean = np.mean(train_latents,axis=0)
train_latents_std = np.std(train_latents,axis=0)
train_latents_whitened = (train_latents - train_latents_mean) / train_latents_std
test_latents_whitened = (test_latents - train_latents_mean) / train_latents_std
new_w = test_latents @ reg_w
new_w_whitened = test_latents_whitened @ reg_w


fmri_train = np.load(f'data/nsd_preproc/sub-{sub:02d}/train_fmriavg_nsdgeneral.npy')
fmri_test = np.load(f'data/nsd_preproc/sub-{sub:02d}/test_fmriavg_nsdgeneral.npy')
fmri_train = fmri_train.reshape(fmri_train.shape[0],-1)
fmri_test = fmri_test.reshape(fmri_test.shape[0],-1)
print(fmri_train.shape, fmri_test.shape)
norm_mean_train = np.mean(fmri_train, axis=0)
norm_scale_train = np.std(fmri_train, axis=0, ddof=1)
fmri_train_whitened = (fmri_train - norm_mean_train) / norm_scale_train
fmri_test_whitened = (fmri_test - norm_mean_train) / norm_scale_train

pred_latents = np.load(f'cache/nsd_preproc/predicted_embeddings/sub-{sub:02d}/regress_vdvae.npy')
pred_latents_whitened = (pred_latents - train_latents_mean) / train_latents_std
new_w_pred = pred_latents @ reg_w
new_w_pred_whitened = pred_latents_whitened @ reg_w
print(new_w_pred.shape, new_w_pred_whitened.shape)

# Path to the folder containing the images
image_folder = f'results/nsd_preproc/sub-{sub:02d}/vdvae'

# Load images from the folder
# image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
image_files = [f"{i}.png" for i in range(982)]
# images = [cv2.imread(os.path.join(image_folder, f)) for f in image_files]
pred_images = [cv2.cvtColor(cv2.imread(os.path.join(image_folder, f)), cv2.COLOR_BGR2RGB) for f in image_files]


# Convert the list of images to a NumPy array
pred_images = np.array(pred_images)

# Verify the shape and data type of the images
print(f"Original images shape: {pred_images.shape}, dtype: {pred_images.dtype}")


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
    return hsv_color[0][0][0]

# def calculate_spatial_frequency(image):
#     # Convert image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
#     # Compute the 2D Fourier Transform of the image
#     f_transform = np.fft.fft2(gray_image)
#     f_transform_shifted = np.fft.fftshift(f_transform)
    
#     # Compute the magnitude spectrum
#     magnitude_spectrum = np.abs(f_transform_shifted)
    
#     # Calculate the spatial frequency as the mean of the magnitude spectrum
#     spatial_frequency = np.mean(magnitude_spectrum)
    
#     return spatial_frequency

def calculate_spatial_frequency(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Compute the 2D Fourier Transform of the image
    f_transform = np.fft.fft2(gray_image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)
    
    # Create a grid of frequencies
    rows, cols = gray_image.shape
    crow, ccol = rows // 2 , cols // 2
    u = np.arange(-crow, crow)
    v = np.arange(-ccol, ccol)
    U, V = np.meshgrid(u, v)
    
    # Calculate the distance from the origin in the frequency domain
    D = np.sqrt(U**2 + V**2)
    
    # Calculate the weighted mean of the magnitude spectrum
    spatial_frequency = np.sum(D * magnitude_spectrum) / np.sum(magnitude_spectrum)
    
    return spatial_frequency

# Verify the shape and data type of the images
print(f"Original images shape: {pred_images.shape}, dtype: {pred_images.dtype}")

# Preprocess images and calculate the average color for each image
preprocessed_images = [preprocess_image(image) for image in pred_images]
average_colors = [calculate_average_color(image) for image in preprocessed_images]

# Calculate luminance and hue for each average color
luminances = [calculate_luminance(color) for color in average_colors]
hues = [calculate_hue(color) for color in average_colors]
# Calculate spatial frequency for each preprocessed image
spatial_frequencies = [calculate_spatial_frequency(image) for image in preprocessed_images]

# Get the argsort indexes for luminance and hue
luminance_argsort = np.argsort(luminances)
hue_argsort = np.argsort(hues)
spatial_frequency_argsort = np.argsort(spatial_frequencies)

def calculate_color_balance(image):
    red_mean = np.mean(image[:, :, 0])
    green_mean = np.mean(image[:, :, 1])
    blue_mean = np.mean(image[:, :, 2])
    return red_mean, green_mean, blue_mean

def balance_color_groups(high_freq_images, low_freq_images, tolerance=1e2):
    def balance_group(images):
        indexes = list(range(len(images)))  # List of indexes
        while True:
            red_means = [calculate_color_balance(images[i])[0] for i in indexes]
            green_means = [calculate_color_balance(images[i])[1] for i in indexes]
            blue_means = [calculate_color_balance(images[i])[2] for i in indexes]
            red_mean = np.mean(red_means)
            green_mean = np.mean(green_means)
            blue_mean = np.mean(blue_means)
            
            if (abs(red_mean - green_mean) < tolerance and
                abs(green_mean - blue_mean) < tolerance and
                abs(red_mean - blue_mean) < tolerance):  # Tolerance for balance
                break
            
            # Find the index that contributes most to the imbalance
            imbalances = [(i, max(abs(calculate_color_balance(images[i])[0] - calculate_color_balance(images[i])[1]),
                                  abs(calculate_color_balance(images[i])[1] - calculate_color_balance(images[i])[2]),
                                  abs(calculate_color_balance(images[i])[0] - calculate_color_balance(images[i])[2]))) for i in indexes]
            imbalances.sort(key=lambda x: x[1], reverse=True)
            
            # Remove the most unbalanced index
            indexes.remove(imbalances[0][0])
        
        return indexes
    
    # Balance each group
    balanced_high_freq_indexes = balance_group(high_freq_images)
    balanced_low_freq_indexes = balance_group(low_freq_images)
    
    # Ensure both groups have the same number of images
    min_length = min(len(balanced_high_freq_indexes), len(balanced_low_freq_indexes))
    balanced_high_freq_indexes = balanced_high_freq_indexes[:min_length]
    balanced_low_freq_indexes = balanced_low_freq_indexes[:min_length]
    
    return balanced_high_freq_indexes, balanced_low_freq_indexes

high_freq_images = pred_images[spatial_frequency_argsort][-400:]
low_freq_images = pred_images[spatial_frequency_argsort][:400]
# balanced_high_freq, balanced_low_freq = balance_color_groups(high_freq_images, low_freq_images)
balanced_high_freq_ids, balanced_low_freq_ids = balance_color_groups(high_freq_images, low_freq_images, tolerance=1e1)
# balanced_high_freq = pred_images[spatial_frequency_argsort][-400:][balanced_high_freq_ids]
# balanced_low_freq = pred_images[spatial_frequency_argsort][:400][balanced_low_freq_ids]

map1 = new_w_pred_whitened[spatial_frequency_argsort][-400:][balanced_high_freq_ids].mean(axis=0) # Textured
map2 = new_w_pred_whitened[spatial_frequency_argsort][:400][balanced_low_freq_ids].mean(axis=0) # Smooth

# Get a random fMRI volumn and substitute its voxels with the patterns
betas = load_img(f'data/nsd_preproc/subj{sub:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz')
betas_trial = index_img(betas, 0)
roi_dir = f'data/nsd_preproc/subj{sub:02d}/func1pt8mm/roi/'
mask_filename = 'nsdgeneral.nii.gz'
mask = nib.load(roi_dir+mask_filename).get_fdata()
betas_trial_masked = betas_trial.get_fdata()

os.makedirs(f'results/nsd_preproc/sub-{sub:02d}/vdvae-texture_patterns/func1pt8mm/', exist_ok=True)
os.makedirs(f'cache/nsd_preproc/predicted_patterns/vdvae-texture_patterns/sub-{sub:02d}/func1pt8mm/', exist_ok=True)
maps = [map1, map2]
pattern_names = ['textured', 'smooth']
for i_map, map in enumerate(maps):
    # plot subject-native pattern
    betas_trial_masked[mask<1] = 0
    betas_trial_masked[mask==1] = map
    vol_data = cortex.Volume(np.moveaxis(np.moveaxis(betas_trial.get_fdata(),0,-1),0,1), f'subj{sub:02d}', 'full', cmap='twilight', vmin=-0.4, vmax=0.4)
    fig = plt.figure(dpi=50) # 100
    cortex.quickflat.make_figure(vol_data, recache=1, fig=fig)
    plt.title(f'{pattern_names[i_map].capitalize()} Pattern - NSD Subject {sub} 1.8mm')
    plt.savefig(f'results/nsd_preproc/sub-{sub:02d}/vdvae-texture_patterns/func1pt8mm/{pattern_names[i_map]}_pattern.png', dpi=300)    
    plt.close()

    pattern_image = nib.Nifti1Image(betas_trial_masked, affine=betas_trial.affine, header=betas_trial.header)
    nib.save(pattern_image, f'cache/nsd_preproc/predicted_patterns/vdvae-texture_patterns/sub-{sub:02d}/func1pt8mm/{pattern_names[i_map]}_pattern.nii.gz')