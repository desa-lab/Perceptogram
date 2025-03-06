import numpy as np
import pickle
import cv2
from scipy.interpolate import griddata
from scipy.cluster.hierarchy import fcluster, linkage, leaves_list
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
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

avg_blue_map_negative = np.load('cache/thingseeg2_preproc/predicted_patterns/ica-color_patterns/avg_blue_pattern_negative.npy')
avg_red_map_negative = np.load('cache/thingseeg2_preproc/predicted_patterns/ica-color_patterns/avg_red_pattern_negative.npy')

# Create figure
fig, axs = plt.subplots(30, 1, figsize=(20, 20), dpi=60)

for i_time in range(0, 60, 2):
    row = i_time // 2
    # axs[row, column].imshow(np.random.rand(10, 10), cmap='gray')
    axs[row].set_xticks([])
    axs[row].set_yticks([])

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, avg_blue_map_negative[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, avg_blue_map_negative[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    axs[row].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Blues', vmin=-0, vmax=0.02)

plt.subplots_adjust(hspace=0.01)
plt.savefig('results/thingseeg2_preproc/blue_pattern_negative_avg.png')

fig, axs = plt.subplots(30, 1, figsize=(20, 20), dpi=60)

for i_time in range(0, 60, 2):
    row = i_time // 2
    # axs[row, column].imshow(np.random.rand(10, 10), cmap='gray')
    axs[row].set_xticks([])
    axs[row].set_yticks([])

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, avg_red_map_negative[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, avg_red_map_negative[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    axs[row].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Reds', vmin=-0, vmax=0.02)

plt.subplots_adjust(hspace=0.01)
plt.savefig('results/thingseeg2_preproc/red_pattern_negative_avg.png')

i_time = 12
# Interpolate the EEG data over the grid
linear_interp = griddata(channel_locs, avg_blue_map_negative[i_time], (grid_x2, grid_y2), method='linear')
nan_mask = np.isnan(linear_interp)
grid_z = griddata(channel_locs, avg_blue_map_negative[i_time], (grid_x, grid_y), method='nearest')
grid_z[nan_mask] = np.nan
grid_z = np.ma.masked_where((grid_z == 0), grid_z)
# Plot the topoplot
plt.figure(figsize=(4, 1.5), dpi=120)
plt.imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Blues', vmin=-0, vmax=0.02)


plt.scatter(channel_locs[:, 0], channel_locs[:, 1], c='k', s=12, edgecolor='k')
xtick_labels = ['Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2']
for i, (x, y) in enumerate(channel_locs):
    plt.text(x, y+0.03, xtick_labels[i], color='k', ha='center', va='center')
plt.title(f'Blue Pattern at {i_time*10} ms')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig(f'results/thingseeg2_preproc/blue_pattern_negative_avg_{i_time*10}ms.png')

i_time = 12
# Interpolate the EEG data over the grid
linear_interp = griddata(channel_locs, avg_red_map_negative[i_time], (grid_x2, grid_y2), method='linear')
nan_mask = np.isnan(linear_interp)
grid_z = griddata(channel_locs, avg_red_map_negative[i_time], (grid_x, grid_y), method='nearest')
grid_z[nan_mask] = np.nan
grid_z = np.ma.masked_where((grid_z == 0), grid_z)
# Plot the topoplot
plt.figure(figsize=(4, 1.5), dpi=120)
plt.imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Reds', vmin=-0, vmax=0.02)


plt.scatter(channel_locs[:, 0], channel_locs[:, 1], c='k', s=12, edgecolor='k')
xtick_labels = ['Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2']
for i, (x, y) in enumerate(channel_locs):
    plt.text(x, y+0.03, xtick_labels[i], color='k', ha='center', va='center')
plt.title(f'Red Pattern at {i_time*10} ms')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig(f'results/thingseeg2_preproc/red_pattern_negative_avg_{i_time*10}ms.png')