import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

avg_busy_map = np.load('cache/thingseeg2_preproc/misc/avg_busy_map.npy')
avg_smooth_map = np.load('cache/thingseeg2_preproc/misc/avg_smooth_map.npy')
avg_warm_map = np.load('cache/thingseeg2_preproc/misc/avg_warm_map.npy')
avg_cold_map = np.load('cache/thingseeg2_preproc/misc/avg_cold_map.npy')
avg_bright_map = np.load('cache/thingseeg2_preproc/misc/avg_bright_map.npy') # since the original map was inverted, here we are inverting it back
avg_dark_map = np.load('cache/thingseeg2_preproc/misc/avg_dark_map.npy') # since the original map was inverted, here we are inverting it back
avg_food_map = np.load('cache/thingseeg2_preproc/misc/avg_food_map.npy')
avg_animal_map = np.load('cache/thingseeg2_preproc/misc/avg_animal_map.npy')
avg_others_map = np.load('cache/thingseeg2_preproc/misc/avg_others_map.npy')

# set positive values to 0
avg_busy_map[avg_busy_map > 0] = 0
avg_smooth_map[avg_smooth_map > 0] = 0
avg_warm_map[avg_warm_map > 0] = 0
avg_cold_map[avg_cold_map > 0] = 0
# avg_bright_map[avg_bright_map > 0] = 0
# avg_dark_map[avg_dark_map > 0] = 0

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

# Example EEG data (replace with your actual data)
# food_map_positive = np.random.randn(30, 17)  # Replace with your actual EEG data
# food_map_negative = np.random.randn(30, 17)  # Replace with your actual EEG data

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

# Ensure the lengths match
# assert len(food_map_positive[0]) == len(channel_locs), "Mismatch between number of EEG data points and channel locations"

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

# Create figure
# fig, axs = plt.subplots(30, 6, figsize=(24, 40), dpi=60)
fig, axs = plt.subplots(30, 9, figsize=(36, 40), dpi=60)

# column = 0
# diff_map = avg_bright_map
# for i_time in range(0, 60, 2):
#     # if i_time != 12:
#     #     continue
#     row = i_time // 2
#     # print(row)
#     axs[row, column].set_xticks([])
#     axs[row, column].set_yticks([])

#     # Interpolate the EEG data over the grid
#     linear_interp = griddata(channel_locs, diff_map[i_time], (grid_x2, grid_y2), method='linear')
#     nan_mask = np.isnan(linear_interp)
#     grid_z = griddata(channel_locs, diff_map[i_time], (grid_x, grid_y), method='nearest')
#     grid_z[nan_mask] = np.nan
#     # grid_z = np.ma.masked_where((grid_z == 0), grid_z)
#     # Plot the topoplot
#     axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Greys', vmin=-0.2, vmax=0.2)
# column = 1
# diff_map = -avg_dark_map
# for i_time in range(0, 60, 2):
#     # if i_time != 12:
#     #     continue
#     row = i_time // 2
#     # print(row)
#     axs[row, column].set_xticks([])
#     axs[row, column].set_yticks([])

#     # Interpolate the EEG data over the grid
#     linear_interp = griddata(channel_locs, diff_map[i_time], (grid_x2, grid_y2), method='linear')
#     nan_mask = np.isnan(linear_interp)
#     grid_z = griddata(channel_locs, diff_map[i_time], (grid_x, grid_y), method='nearest')
#     grid_z[nan_mask] = np.nan
#     # grid_z = np.ma.masked_where((grid_z == 0), grid_z)
#     # Plot the topoplot
#     axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Greys', vmin=-0.2, vmax=0.2)
column = 0
diff_map = avg_bright_map
diff_map_positive = np.maximum(diff_map, 0)
diff_map_negative = - np.minimum(diff_map, 0)
for i_time in range(0, 60, 2):
    row = i_time // 2
    axs[row, column].set_xticks([])
    axs[row, column].set_yticks([])

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_positive[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_positive[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Reds', vmin=0, vmax=0.8) #0.3

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_negative[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_negative[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Blues', vmin=0, vmax=0.8)
column = 1
diff_map = avg_dark_map
diff_map_positive = np.maximum(diff_map, 0)
diff_map_negative = - np.minimum(diff_map, 0)
for i_time in range(0, 60, 2):
    row = i_time // 2
    axs[row, column].set_xticks([])
    axs[row, column].set_yticks([])

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_positive[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_positive[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Reds', vmin=0, vmax=0.8)

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_negative[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_negative[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Blues', vmin=0, vmax=0.8)
column = 2
diff_map = -avg_warm_map
diff_map_positive = np.maximum(diff_map, 0)
diff_map_negative = - np.minimum(diff_map, 0)
for i_time in range(0, 60, 2):
    row = i_time // 2
    axs[row, column].set_xticks([])
    axs[row, column].set_yticks([])

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_positive[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_positive[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Reds', vmin=0, vmax=0.7)

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_negative[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_negative[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Blues', vmin=0, vmax=0.7)
column = 3
diff_map = avg_cold_map
diff_map_positive = np.maximum(diff_map, 0)
diff_map_negative = - np.minimum(diff_map, 0)
for i_time in range(0, 60, 2):
    row = i_time // 2
    axs[row, column].set_xticks([])
    axs[row, column].set_yticks([])

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_positive[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_positive[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Reds', vmin=0, vmax=0.7)

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_negative[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_negative[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Blues', vmin=0, vmax=0.7)
column = 4
diff_map = -avg_busy_map
diff_map_positive = np.maximum(diff_map, 0)
diff_map_negative = - np.minimum(diff_map, 0)
for i_time in range(0, 60, 2):
    row = i_time // 2
    axs[row, column].set_xticks([])
    axs[row, column].set_yticks([])

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_positive[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_positive[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Oranges', vmin=0, vmax=10)

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_negative[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_negative[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Purples', vmin=0, vmax=10)
column = 5
diff_map = avg_smooth_map
diff_map_positive = np.maximum(diff_map, 0)
diff_map_negative = - np.minimum(diff_map, 0)
for i_time in range(0, 60, 2):
    row = i_time // 2
    axs[row, column].set_xticks([])
    axs[row, column].set_yticks([])

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_positive[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_positive[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Oranges', vmin=0, vmax=10)

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_negative[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_negative[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Purples', vmin=0, vmax=10)

column = 6
diff_map = avg_food_map
diff_map_positive = np.maximum(diff_map, 0)
diff_map_negative = - np.minimum(diff_map, 0)
for i_time in range(0, 60, 2):
    row = i_time // 2
    axs[row, column].set_xticks([])
    axs[row, column].set_yticks([])

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_positive[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_positive[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Reds', vmin=0, vmax=1)

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_negative[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_negative[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Blues', vmin=0, vmax=1)

column = 7
diff_map = avg_animal_map
diff_map_positive = np.maximum(diff_map, 0)
diff_map_negative = - np.minimum(diff_map, 0)
for i_time in range(0, 60, 2):
    row = i_time // 2
    axs[row, column].set_xticks([])
    axs[row, column].set_yticks([])

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_positive[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_positive[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Greens', vmin=0, vmax=1)
column = 8
diff_map = avg_others_map
diff_map_positive = np.maximum(diff_map, 0)
diff_map_negative = - np.minimum(diff_map, 0)
for i_time in range(0, 60, 2):
    row = i_time // 2
    axs[row, column].set_xticks([])
    axs[row, column].set_yticks([])

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_positive[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_positive[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Blues', vmin=0, vmax=1)

    # Interpolate the EEG data over the grid
    linear_interp = griddata(channel_locs, diff_map_negative[i_time], (grid_x2, grid_y2), method='linear')
    nan_mask = np.isnan(linear_interp)
    grid_z = griddata(channel_locs, diff_map_negative[i_time], (grid_x, grid_y), method='nearest')
    grid_z[nan_mask] = np.nan
    grid_z = np.ma.masked_where((grid_z == 0), grid_z)
    # Plot the topoplot
    axs[row, column].imshow(grid_z.T, extent=(grid_x.min()- extension * x_range, grid_x.max()+ extension * x_range, grid_y.min()- extension * y_range, grid_y.max()+ 0.2 * y_range), origin='lower', cmap='Blues', vmin=0, vmax=1)

plt.subplots_adjust(hspace=0.0)
plt.tight_layout()
os.makedirs('results/thingseeg2_preproc/avg', exist_ok=True)
plt.savefig('results/thingseeg2_preproc/avg/avg_patterns_all.png', bbox_inches='tight')