import numpy as np
import pickle
import cv2
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 10-10 system channel locations (x, y coordinates) for 17 channels
channel_locs = np.array([[-0.00170945, -0.04521299],
       [-0.05503823, -0.0442103 ],
       [-0.07445796, -0.04212316],
       [-0.03157356, -0.08056835],
       [-0.00206025, -0.08278299],
       [ 0.0276831 , -0.08048884],
       [ 0.05363602, -0.04433452],
       [ 0.07103247, -0.04225998],
       [-0.03065362, -0.04492739],
       [-0.06929984, -0.04322697],
       [-0.05694862, -0.06592325],
       [-0.0386246 , -0.06736158],
       [-0.00189982, -0.0680541 ],
       [ 0.03466779, -0.06766214],
       [ 0.05355732, -0.06641694],
       [ 0.06586062, -0.04333852],
       [ 0.02988639, -0.04503254]]) * 6
# Calculate the min and max coordinates
min_x, min_y = channel_locs.min(axis=0)
max_x, max_y = channel_locs.max(axis=0)
extension1 = 0.15
extension2 = 0
extension = 0
# Extend the grid range by 10%
x_range = max_x - min_x
y_range = max_y - min_y
grid_x, grid_y = np.mgrid[(min_x - extension1 * x_range):(max_x + extension1 * x_range):1000j, 
                          (min_y - extension1 * y_range):(max_y + extension1 * y_range):1000j]
grid_x2, grid_y2 = np.mgrid[(min_x - extension2 * x_range):(max_x + extension2 * x_range):1000j, 
                          (min_y - extension2 * y_range):(max_y + extension2 * y_range):1000j]


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

train_latents= np.load('cache/thingseeg2_extracted_embeddings/train_pca1k.npy')
test_latents = np.load('cache/thingseeg2_extracted_embeddings/test_pca1k.npy')

os.makedirs('cache/thingseeg2_preproc/predicted_patterns/pca-brightness_patterns', exist_ok=True)
# Create figure
fig, axs = plt.subplots(30, 10, figsize=(40, 40), dpi=60)

all_diff_maps = []
all_bright_maps = []
all_dark_maps = []
for sub in tqdm(range(1, 11), total=10):
    os.makedirs(f'cache/thingseeg2_preproc/predicted_patterns/pca-brightness_patterns/sub-{sub:02d}', exist_ok=True)

    with open(f'cache/thingseeg2_preproc/regression_weights/sub-{sub:02d}/regress-encode_pca1k_weights.pkl',"rb") as f:
        datadict = pickle.load(f)
        reg_w = datadict['weight'].T
        reg_b = datadict['bias']

    train_latents_mean = np.mean(train_latents,axis=0)
    train_latents_std = np.std(train_latents,axis=0)
    train_latents_whitened = (train_latents - train_latents_mean) / train_latents_std
    test_latents_whitened = (test_latents - train_latents_mean) / train_latents_std
    new_w = test_latents @ reg_w
    new_w_whitened = test_latents_whitened @ reg_w

    eeg_train = np.load(f'data/thingseeg2_preproc/sub-{sub:02d}/train_thingseeg2_avg.npy')
    eeg_test = np.load(f'data/thingseeg2_preproc/sub-{sub:02d}/test_thingseeg2_avg.npy')
    eeg_train = eeg_train.reshape(eeg_train.shape[0],-1)
    eeg_test = eeg_test.reshape(eeg_test.shape[0],-1)
    # print(eeg_train.shape, eeg_test.shape)
    norm_mean_train = np.mean(eeg_train, axis=0)
    norm_scale_train = np.std(eeg_train, axis=0, ddof=1)
    eeg_train_whitened = (eeg_train - norm_mean_train) / norm_scale_train
    eeg_test_whitened = (eeg_test - norm_mean_train) / norm_scale_train

    pred_latents = np.load(f'cache/thingseeg2_preproc/predicted_embeddings/sub-{sub:02d}/regress_pca1k.npy')
    pred_latents_whitened = (pred_latents - train_latents_mean) / train_latents_std
    new_w_pred = pred_latents @ reg_w
    new_w_pred_whitened = pred_latents_whitened @ reg_w
    # print(new_w_pred.shape, new_w_pred_whitened.shape)


    # Path to the folder containing the images
    image_folder = f'results/thingseeg2_preproc/sub-{sub:02d}/pca1k'
    # Load images from the folder
    image_files = [f"{i:03d}.png" for i in range(200)]
    images = [cv2.cvtColor(cv2.imread(os.path.join(image_folder, f)), cv2.COLOR_BGR2RGB) for f in image_files]
    # Convert the list of images to a NumPy array
    images = np.array(images)
    # Preprocess images and calculate the average color for each image
    preprocessed_images = [preprocess_image(image) for image in images]
    average_colors = [calculate_average_color(image) for image in preprocessed_images]
    # Calculate luminance and hue for each average color
    luminances = [calculate_luminance(color) for color in average_colors]
    hues = [calculate_hue(color) for color in average_colors]
    # Get the argsort indexes for luminance and hue
    luminance_argsort = np.argsort(luminances)
    hue_argsort = np.argsort(hues)

    bright_map = new_w_pred[luminance_argsort][-70:].mean(axis=0).reshape(17, 80).T
    dark_map = new_w_pred[luminance_argsort][:70].mean(axis=0).reshape(17, 80).T
    np.save(f'cache/thingseeg2_preproc/predicted_patterns/pca-brightness_patterns/sub-{sub:02d}/bright_pattern.npy', bright_map)
    np.save(f'cache/thingseeg2_preproc/predicted_patterns/pca-brightness_patterns/sub-{sub:02d}/dark_pattern.npy', dark_map)
    all_bright_maps.append(bright_map)
    all_dark_maps.append(dark_map)

    column = sub - 1

    for i_time in range(0, 60, 2):
        row = i_time // 2
        axs[row, column].set_xticks([])
        axs[row, column].set_yticks([])

        # Interpolate the EEG data over the grid
        linear_interp = griddata(channel_locs, bright_map[i_time], (grid_x2, grid_y2), method='linear')
        nan_mask = np.isnan(linear_interp)
        grid_z = griddata(channel_locs, bright_map[i_time], (grid_x, grid_y), method='nearest')
        grid_z[nan_mask] = np.nan
        grid_z = np.ma.masked_where((grid_z == 0), grid_z)
        axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='RdBu_r', vmin=-0.4, vmax=0.4)

# all_diff_maps = np.array(all_diff_maps)
# avg_diff_map = all_diff_maps.mean(axis=0)
all_bright_maps = np.array(all_bright_maps)
avg_bright_map = all_bright_maps.mean(axis=0)
all_dark_maps = np.array(all_dark_maps)
avg_dark_map = all_dark_maps.mean(axis=0)

np.save('cache/thingseeg2_preproc/predicted_patterns/pca-brightness_patterns/avg_bright_pattern.npy', avg_bright_map)
np.save('cache/thingseeg2_preproc/predicted_patterns/pca-brightness_patterns/avg_dark_pattern.npy', avg_dark_map)

plt.subplots_adjust(hspace=0.0)
plt.tight_layout()
plt.savefig('results/thingseeg2_preproc/bright_pattern.png')


# Create figure
fig, axs = plt.subplots(30, 10, figsize=(40, 40), dpi=60)

for sub in tqdm(range(1, 11), total=10):

    dark_map = all_dark_maps[sub-1]

    column = sub - 1

    for i_time in range(0, 60, 2):
        row = i_time // 2
        axs[row, column].set_xticks([])
        axs[row, column].set_yticks([])

        # Interpolate the EEG data over the grid
        linear_interp = griddata(channel_locs, dark_map[i_time], (grid_x2, grid_y2), method='linear')
        nan_mask = np.isnan(linear_interp)
        grid_z = griddata(channel_locs, dark_map[i_time], (grid_x, grid_y), method='nearest')
        grid_z[nan_mask] = np.nan
        grid_z = np.ma.masked_where((grid_z == 0), grid_z)
        axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='RdBu_r', vmin=-0.4, vmax=0.4)

plt.subplots_adjust(hspace=0.0)
plt.tight_layout()
plt.savefig('results/thingseeg2_preproc/dark_pattern.png')