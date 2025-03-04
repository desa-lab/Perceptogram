import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import os
import sys
import numpy as np
import scipy.io as spio
import scipy as sp
from PIL import Image
from scipy.stats import pearsonr,binom,linregress
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-ordered", "--ordered_by_performance",help="Ordered by performance",default=False)
parser.add_argument("-param", "--param",help="Transfer Learning Parameter",default='3000avg_transferred_from2')
args = parser.parse_args()
sub=int(args.sub)
param=args.param
ordered_by_performance = args.ordered_by_performance

recon_dir = f'results/thingseeg2_preproc/sub-{sub:02d}/'
image_ids = list(range(200))
if ordered_by_performance:

    feats_dir = f'cache/thingseeg2_preproc/eval_features/sub-{sub:02d}/versatile_diffusion_{param}'
    test_feats_dir = 'cache/thingseeg2_test_images_eval_features'

    def pairwise_corr_all(ground_truth, predictions):
        r = np.corrcoef(ground_truth, predictions)#cosine_similarity(ground_truth, predictions)#
        r = r[:len(ground_truth), len(ground_truth):]  # rows: groundtruth, columns: predicitons
        #print(r.shape)
        # congruent pairs are on diagonal
        congruents = np.diag(r)
        #print(congruents)
        
        # for each column (predicition) we should count the number of rows (groundtruth) that the value is lower than the congruent (e.g. success).
        success = r < congruents
        success_cnt = np.sum(success, 0)
        
        # note: diagonal of 'success' is always zero so we can discard it. That's why we divide by len-1
        perf = np.mean(success_cnt) / (len(ground_truth)-1)
        p = 1 - binom.cdf(perf*len(ground_truth)*(len(ground_truth)-1), len(ground_truth)*(len(ground_truth)-1), 0.5)
        
        return perf, p

    def pairwise_corr_individuals(ground_truth, predictions):
        r = np.corrcoef(ground_truth, predictions)#cosine_similarity(ground_truth, predictions)#
        r = r[:len(ground_truth), len(ground_truth):]  # rows: groundtruth, columns: predicitons
        #print(r.shape)
        # congruent pairs are on diagonal
        congruents = np.diag(r)
        #print(congruents)
        
        # for each column (predicition) we should count the number of rows (groundtruth) that the value is lower than the congruent (e.g. success).
        success = r < congruents
        success_cnt = np.sum(success, 0)
        
        # note: diagonal of 'success' is always zero so we can discard it. That's why we divide by len-1
        perf = success_cnt / (len(ground_truth)-1)
        # p = 1 - binom.cdf(perf*len(ground_truth)*(len(ground_truth)-1), len(ground_truth)*(len(ground_truth)-1), 0.5)
        
        return perf

    net_list = [
        ('inceptionv3','avgpool'),
        ('clip','final'),
        ('alexnet',2),
        ('alexnet',5),
        ('efficientnet','avgpool'),
        ('swav','avgpool')
        ]
    
    num_test = 200
    distance_fn = sp.spatial.distance.correlation
    pairwise_corrs = []
    for (net_name,layer) in net_list:
        file_name = '{}/{}_{}.npy'.format(test_feats_dir,net_name,layer)
        gt_feat = np.load(file_name)
        
        file_name = '{}/{}_{}.npy'.format(feats_dir,net_name,layer)
        eval_feat = np.load(file_name)
        
        gt_feat = gt_feat.reshape((len(gt_feat),-1))
        eval_feat = eval_feat.reshape((len(eval_feat),-1))

        print(net_name,layer)
        if net_name in ['efficientnet','swav']:
            print('distance: ',np.array([distance_fn(gt_feat[i],eval_feat[i]) for i in range(num_test)]).mean())
        else:
            # pairwise_corrs.append(pairwise_corr_all(gt_feat[:num_test],eval_feat[:num_test])[0])
            pairwise_corrs.append(pairwise_corr_individuals(gt_feat[:num_test],eval_feat[:num_test]))
            print('pairwise corr: ',pairwise_corr_all(gt_feat[:num_test],eval_feat[:num_test])[0])

    inception_inds = np.argsort(pairwise_corrs[0])[::-1]
    clip_inds = np.argsort(pairwise_corrs[1])[::-1]
    alexnet2_inds = np.argsort(pairwise_corrs[2])[::-1]
    alexnet5_inds = np.argsort(pairwise_corrs[3])[::-1]
    image_ids = clip_inds


def load_test_image(image_id):
    return mpimg.imread(f'data/thingseeg2_metadata/test_images_direct/{image_id}.png')
def load_versatile_diffusion_image(image_id):
    return mpimg.imread(recon_dir + f'versatile_diffusion_{param}/{image_id}.png')
def load_vdvae_image(image_id):
    return mpimg.imread(recon_dir + f'vdvae/{image_id}.png')
# Load stimulus images
stimulus_images = [load_test_image(image_id) for image_id in image_ids]
# Load reconstructed images
reconstructed_images = [load_versatile_diffusion_image(image_id) for image_id in image_ids]
# Create figure
fig, axs = plt.subplots(20, 20, figsize=(20, 20))

for row in range(10):
    # Add stimulus images to top row
    for i, img in enumerate(stimulus_images[20 * row:20 * row + 20]):
        axs[2 * row, i].imshow(img)
        axs[2 * row, i].axis('off')
        rect = Rectangle((0,0), img.shape[1], img.shape[0], linewidth=5, edgecolor='b', facecolor='none')
        axs[2 * row, i].add_patch(rect)

    # Add reconstructed images to bottom row with a red frame
    for i, img in enumerate(reconstructed_images[20 * row:20 * row + 20]):
        axs[2 * row + 1, i].imshow(img)
        axs[2 * row + 1, i].axis('off')
        # Add a red frame
        rect = Rectangle((0,0), img.shape[1], img.shape[0], linewidth=5, edgecolor='g', facecolor='none')
        axs[2 * row + 1, i].add_patch(rect)

# Reduce the gap between images
plt.subplots_adjust(wspace=0.02, hspace=0.02)

if ordered_by_performance:
    plt.savefig(recon_dir+f'diffusion_recon_plot_ordered_by_performance_{param}.png',bbox_inches='tight')
else:
    plt.savefig(recon_dir+f'diffusion_recon_plot_{param}.png',bbox_inches='tight')

# Load reconstructed images
reconstructed_images = [load_vdvae_image(image_id) for image_id in image_ids]
# Create figure
fig, axs = plt.subplots(20, 20, figsize=(20, 20))

for row in range(10):
    # Add stimulus images to top row
    for i, img in enumerate(stimulus_images[20 * row:20 * row + 20]):
        axs[2 * row, i].imshow(img)
        axs[2 * row, i].axis('off')
        rect = Rectangle((0,0), img.shape[1], img.shape[0], linewidth=5, edgecolor='b', facecolor='none')
        axs[2 * row, i].add_patch(rect)

    # Add reconstructed images to bottom row with a red frame
    for i, img in enumerate(reconstructed_images[20 * row:20 * row + 20]):
        axs[2 * row + 1, i].imshow(img)
        axs[2 * row + 1, i].axis('off')
        # Add a red frame
        rect = Rectangle((0,0), img.shape[1], img.shape[0], linewidth=5, edgecolor='g', facecolor='none')
        axs[2 * row + 1, i].add_patch(rect)

# Reduce the gap between images
plt.subplots_adjust(wspace=0.02, hspace=0.02)

if ordered_by_performance:
    plt.savefig(recon_dir+f'vdvae_recon_plot_ordered_by_performance_{param}.png',bbox_inches='tight')
else:
    plt.savefig(recon_dir+f'vdvae_recon_plot_{param}.png',bbox_inches='tight')