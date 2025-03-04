import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

subs = [1,2,3,4]
duration = [0, 5, 10, 20, 40, 60, 80]
performance = np.zeros((len(duration), len(subs), 8))
for i, dur in enumerate(duration):
    for j, sub in enumerate(subs):
        if dur == 80:
            performance[i, j] = np.load(f'results/thingseeg2_preproc/sub-{sub:02d}/performances_versatile_diffusion.npy')
        else:
            performance[i, j] = np.load(f'results/thingseeg2_preproc/sub-{sub:02d}/performances_versatile_diffusion_16540avg_dur{dur}.npy')
performance_mean = np.mean(performance, axis=1)
performance_std = np.std(performance, axis=1)

import matplotlib.cm as cm

# Set the style of the plot
plt.style.use("ggplot")

# Define the figure size
plt.figure(figsize=(9, 6))

# Define the bar width
bar_width = 0.12

# Define the x values for the bar plots
x = np.arange(len(performance_mean[0]))

# Define the colors for each duration using matplotlib's colormap
colors = cm.get_cmap('tab20c', 7)

# Plot the bar plots with error bars
for i in range(7):
    plt.bar(x + i*bar_width, performance_mean[i], color=colors(i), width=bar_width, label=f'{duration[i]*10}ms', yerr=performance_std[i], capsize=4)

# Define the labels for the x-axis
labels = ['PixCorr↑', 'SSIM↑', 'Alex(2)↑', 'Alex(5)↑', 'Incep↑', 'CLIP↑', 'Eff↓', 'SwAV↓']  # replace with your actual labels

# Set the current tick locations and labels of the x-axis
plt.xticks(x + bar_width*3.5, labels, fontsize=12)  # Adjusted to center the labels
plt.yticks(fontsize=12)
plt.ylim(0, 1)

# Add a legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

plt.title('Performance Across Durations', fontsize=16)
plt.ylabel('Performance', fontsize=14)
plt.xlabel('Metric', fontsize=14)

# Display the plot
plt.tight_layout()

plt.savefig('results/thingseeg2_preproc/fig_across_duration.png')
plt.savefig('results/thingseeg2_preproc/fig_across_duration.svg', format='svg')