import os
import sys
import numpy as np
# import h5py
import scipy.io as spio
# import nibabel as nib
import scipy as sp
from PIL import Image



import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-source", "--transfer_source",help="Transfer source",default=2)
parser.add_argument("-size", "--size",help="Size",default=3000)
parser.add_argument('-avg', '--average', help='Number of averages', default='')
parser.add_argument("-transfering", "--transfering",help="Transfering",default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("-method", "--method",help="Transfer Learning Method",default='augment') # augment or transfer
# parser.add_argument("-param", "--param",help="Transfer Learning Parameter",default='3000avg_transfered_from2')
args = parser.parse_args()
sub=int(args.sub)
transfer_source=int(args.transfer_source) if args.transfer_source != 'other' else 'other'
common_size=int(args.size)
average=args.average
transferred= 'transferred' if args.transfering else 'nontransferred'
method=args.method
if method == 'augment':
    if average != '':
        param=f'{common_size}avg{average}_{transferred}_from{transfer_source}'
    else:
        param=f'{common_size}avg_{transferred}_from{transfer_source}'
elif method == 'transfer':
    if args.transfering:
        param=f'{common_size}avg_{transferred}_to{transfer_source}'
    else:
        param=f'avg_{transferred}_to{transfer_source}'
# param=args.param


from scipy.stats import pearsonr,binom,linregress
import numpy as np
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


net_list = [
    ('alexnet',2),
    ('alexnet',5),
    ('inceptionv3','avgpool'),
    ('clip','final'),
    ('efficientnet','avgpool'),
    ('swav','avgpool')
    ]

performances = []

num_test = 200
test_dir = 'cache/thingseeg2_test_images_eval_features'

feats_dir = f'cache/thingseeg2_preproc/eval_features/sub-{sub:02d}/versatile_diffusion_{param}'
gen_image_dir = f'results/thingseeg2_preproc/sub-{sub:02d}/versatile_diffusion_{param}/'
performances_save_dir = f'results/thingseeg2_preproc/sub-{sub:02d}/performances_versatile_diffusion_{param}.npy'



from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
        
ssim_list = []
pixcorr_list = []


for i in range(200):
    gt_image = Image.open('data/thingseeg2_metadata/test_images_direct/{}.png'.format(i)).resize((512,512)) # either both 512 or both 500
    gen_image = Image.open(gen_image_dir + '{}.png'.format(i))


    gen_image = np.array(gen_image)/255.0
    gt_image = np.array(gt_image)/255.0
    pixcorr_res = np.corrcoef(gt_image.reshape(1,-1), gen_image.reshape(1,-1))[0,1]
    pixcorr_list.append(pixcorr_res)
    gen_image = rgb2gray(gen_image)
    gt_image = rgb2gray(gt_image)
    ssim_res = ssim(gen_image, gt_image, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
    ssim_list.append(ssim_res)
    
ssim_list = np.array(ssim_list)
pixcorr_list = np.array(pixcorr_list)
print('PixCorr: {}'.format(pixcorr_list.mean()))
print('SSIM: {}'.format(ssim_list.mean()))
performances.append(pixcorr_list.mean())
performances.append(ssim_list.mean())

distance_fn = sp.spatial.distance.correlation
pairwise_corrs = []
for (net_name,layer) in net_list:
    file_name = '{}/{}_{}.npy'.format(test_dir,net_name,layer)
    gt_feat = np.load(file_name)
    
    file_name = '{}/{}_{}.npy'.format(feats_dir,net_name,layer)
    eval_feat = np.load(file_name)
    
    gt_feat = gt_feat.reshape((len(gt_feat),-1))
    eval_feat = eval_feat.reshape((len(eval_feat),-1))
    
    print(net_name,layer)
    if net_name in ['efficientnet','swav']:
        print('distance: ',np.array([distance_fn(gt_feat[i],eval_feat[i]) for i in range(num_test)]).mean())
        performances.append(np.array([distance_fn(gt_feat[i],eval_feat[i]) for i in range(num_test)]).mean())
    else:
        pairwise_corrs.append(pairwise_corr_all(gt_feat[:num_test],eval_feat[:num_test])[0])
        print('pairwise corr: ',pairwise_corrs[-1])
        performances.append(pairwise_corrs[-1])
        
performances = np.array(performances)
np.save(performances_save_dir,performances)
