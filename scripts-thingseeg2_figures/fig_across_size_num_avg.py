import matplotlib.pyplot as plt
import numpy as np

sizes = [500, 1000, 2000, 4000, 6000, 10000, 14000, 16540]
avgs = [5, 10, 20, 30, 40, 60, 80]
performance = np.zeros((len(sizes), len(avgs), 8))
sub = 1
for i, size in enumerate(sizes):
    for j, avg in enumerate(avgs):
        if avg == 80 and size == 16540:
            performance[i, j] = np.load(f'results/thingseeg2_preproc/sub-{sub:02d}/performances_versatile_diffusion.npy')
        elif avg == 80:
            performance[i, j] = np.load(f'results/thingseeg2_preproc/sub-{sub:02d}/performances_versatile_diffusion_{size}avg.npy')
        else:
            performance[i, j] = np.load(f'results/thingseeg2_preproc/sub-{sub:02d}/performances_versatile_diffusion_{size}avg{avg}.npy')


# Create a figure
fig, ax = plt.subplots(figsize=(10*0.7, 8*0.7))

# Calculate the edges of the cells for the x and y coordinates
avgs_edges = np.concatenate([[0], avgs])

# Calculate the edges for the sizes
sizes_edges = np.concatenate([[0], sizes])

# Calculate the centers of the cells for the x and y coordinates
avgs_centers = (avgs_edges[:-1] + avgs_edges[1:]) / 2
sizes_centers = (sizes_edges[:-1] + sizes_edges[1:]) / 2

# Create a meshgrid for the x and y coordinates
X, Y = np.meshgrid(avgs_edges, sizes_edges)

# Create the heatmap using pcolormesh
c = ax.pcolormesh(X, Y, performance[:,:,5], cmap='viridis')

# Create colorbar
fig.colorbar(c, ax=ax, label='Color scale')

# Set the labels for the x ticks
ax.set_xticks(avgs_edges)
# ax.set_xticklabels(avgs)
ax.set_xlabel('Number of averaged test trials per image')

# Set the labels for the y ticks
ax.set_yticks(sizes_edges)
ax.set_ylabel('Number of training images')

# Set the title
ax.set_title('CLIP Performance of Subject 1')

# Loop over data dimensions and create text annotations.
for i in range(performance.shape[0]):
    for j in range(performance.shape[1]):
        ax.text(avgs_centers[j], sizes_centers[i], f'{performance[i, j, 5]:.2g}',
                ha="center", va="center", color="w", fontsize=7)

plt.savefig('results/thingseeg2_preproc/fig_CLIP_across_size_num_avg.png')
plt.savefig('results/thingseeg2_preproc/fig_CLIP_across_size_num_avg.svg', format='svg')