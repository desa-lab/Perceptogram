import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

subs = list(range(1,11))
seeds = [1, 2, 4, 5, 6, 7, 8]
performance = np.zeros((len(seeds), len(subs), 8))
for i, seed in enumerate(seeds):
    for j, sub in enumerate(subs):
        performance[i, j] = np.load(f'results/thingseeg2_preproc/sub-{sub:02d}/performances_versatile_diffusion_seed{seed}.npy')
performance_mean = np.mean(performance, axis=0)
performance_std = np.std(performance, axis=0)


# Set the style of the plot
plt.style.use("ggplot")

# Define the figure size
plt.figure(figsize=(9, 6))

# Define the bar width
bar_width = 0.08

# Define the x values for the bar plots
x = np.arange(len(performance_mean[0]))

# Define the colors for each subject using matplotlib's colormap
colors = cm.get_cmap('tab20c', 10)

# Plot the bar plots with error bars
for i in range(10):
    plt.bar(x + i*bar_width, performance_mean[i], color=colors(i), width=bar_width, label=f'subject {i+1}', yerr=performance_std[i], capsize=3)

# Define the labels for the x-axis
labels = ['PixCorr↑', 'SSIM↑', 'Alex(2)↑', 'Alex(5)↑', 'Incep↑', 'CLIP↑', 'Eff↓', 'SwAV↓']  # replace with your actual labels

# Set the current tick locations and labels of the x-axis
plt.xticks(x + bar_width*4.5, labels, fontsize=12)  # Adjusted to center the labels
plt.yticks(fontsize=12)
plt.ylim(0, 1)

# Add a legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

plt.title('Performance Across Subjects', fontsize=16)
plt.ylabel('Performance', fontsize=14)
plt.xlabel('Metric', fontsize=14)

# Display the plot
plt.tight_layout()
plt.savefig('results/thingseeg2_preproc/fig_performance.png')
plt.savefig('results/thingseeg2_preproc/fig_performance.svg', format='svg')