import os
import numpy as np
# import h5py
# import nibabel as nib
from PIL import Image
import numpy as np
# import h5py
# import nibabel as nib

import torch
import torchvision.models as tvmodels
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import clip

import skimage.io as sio
from skimage import data, img_as_float
from skimage.transform import resize as imresize
from skimage.metrics import structural_similarity as ssim
import scipy as sp



import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-size", "--size",help="Size",default=16540)
parser.add_argument('-avg', '--average', help='Number of averages', default='')
parser.add_argument('-duration', '--duration', help='Duration', default=80)
parser.add_argument('-seed', '--seed', help='Random Seed', default=0)
parser.add_argument('-vdvae', '--vdvae', help='Using VDVAE', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('-clipvision', '--clipvision', help='Using Clipvision', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('-cliptext', '--cliptext', help='Using Cliptext', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('-dnn', '--using_dnn', help='Using Deep Neural Netoworks', default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()
sub=int(args.sub)
train_size=int(args.size)
average=args.average
duration=int(args.duration)
seed=int(args.seed)
using_vdvae=args.vdvae
using_clipvision=args.clipvision
using_cliptext=args.cliptext
using_dnn=args.using_dnn
if average != '' or train_size != 16540 or duration != 80:
    param = f'_{train_size}avg{average}_dur{duration}'
else:
    param = ''
if using_dnn:
    param += '_dnn'
if not using_vdvae:
    param += '_novdvae'
if not using_clipvision:
    param += '_noclipvision'
if not using_cliptext:
    param += '_nocliptext'
if seed != 0:
    param += f'_seed{seed}'

images_dir = f'results/thingseeg2_preproc/sub-{sub:02d}/versatile_diffusion{param}'
feats_dir = f'cache/thingseeg2_preproc/eval_features/sub-{sub:02d}/versatile_diffusion{param}/'

if not using_cliptext and not using_clipvision:
    images_dir = f'results/thingseeg2_preproc/sub-{sub:02d}/vdvae'
    feats_dir = f'cache/thingseeg2_preproc/eval_features/sub-{sub:02d}/vdvae/'


if not os.path.exists(feats_dir):
   os.makedirs(feats_dir)

class batch_generator_external_images(Dataset):

    def __init__(self, data_path ='', prefix='', net_name='clip'):
        self.data_path = data_path
        self.prefix = prefix
        self.net_name = net_name
        
        if self.net_name == 'clip':
           self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        else:
           self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_test = 200 # 982
        
    def __getitem__(self,idx):
        img = Image.open('{}/{}{}.png'.format(self.data_path,self.prefix,idx))
        img = T.functional.resize(img,(224,224))
        img = T.functional.to_tensor(img).float()
        img = self.normalize(img)
        return img

    def __len__(self):
        return  self.num_test





global feat_list
feat_list = []
def fn(module, inputs, outputs):
    feat_list.append(outputs.cpu().numpy())

net_list = [
    ('inceptionv3','avgpool'),
    ('clip','final'),
    ('alexnet',2),
    ('alexnet',5),
    ('efficientnet','avgpool'),
    ('swav','avgpool')
    ]

device = 1
net = None
batchsize=64



for (net_name,layer) in net_list:
    feat_list = []
    print(net_name,layer)
    dataset = batch_generator_external_images(data_path=images_dir,net_name=net_name,prefix='')
    loader = DataLoader(dataset,batchsize,shuffle=False)
    
    if net_name == 'inceptionv3': # SD Brain uses this
        net = tvmodels.inception_v3(pretrained=True)
        if layer== 'avgpool':
            net.avgpool.register_forward_hook(fn) 
        elif layer == 'lastconv':
            net.Mixed_7c.register_forward_hook(fn)
            
    elif net_name == 'alexnet':
        net = tvmodels.alexnet(pretrained=True)
        if layer==2:
            net.features[4].register_forward_hook(fn)
        elif layer==5:
            net.features[11].register_forward_hook(fn)
        elif layer==7:
            net.classifier[5].register_forward_hook(fn)
            
    elif net_name == 'clip':
        model, _ = clip.load("ViT-L/14", device='cuda:{}'.format(device))
        net = model.visual
        net = net.to(torch.float32)
        if layer==7:
            net.transformer.resblocks[7].register_forward_hook(fn)
        elif layer==12:
            net.transformer.resblocks[12].register_forward_hook(fn)
        elif layer=='final':
            net.register_forward_hook(fn)
    
    elif net_name == 'efficientnet':
        net = tvmodels.efficientnet_b1(weights=True)
        net.avgpool.register_forward_hook(fn) 
        
    elif net_name == 'swav':
        net = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        net.avgpool.register_forward_hook(fn) 
    net.eval()
    net.cuda(device)    
    
    with torch.no_grad():
        for i,x in enumerate(loader):
            print(i*batchsize)
            x = x.cuda(device)
            _ = net(x)
    if net_name == 'clip':
        if layer == 7 or layer == 12:
            feat_list = np.concatenate(feat_list,axis=1).transpose((1,0,2))
        else:
            feat_list = np.concatenate(feat_list)
    else:   
        feat_list = np.concatenate(feat_list)
    
    
    file_name = '{}/{}_{}.npy'.format(feats_dir,net_name,layer)
    np.save(file_name,feat_list)

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

feats_dir = f'cache/thingseeg2_preproc/eval_features/sub-{sub:02d}/versatile_diffusion{param}'
gen_image_dir = f'results/thingseeg2_preproc/sub-{sub:02d}/versatile_diffusion{param}/'
performances_save_dir = f'results/thingseeg2_preproc/sub-{sub:02d}/performances_versatile_diffusion{param}.npy'

if not using_cliptext and not using_clipvision:
    feats_dir = f'cache/thingseeg2_preproc/eval_features/sub-{sub:02d}/vdvae'
    gen_image_dir = f'results/thingseeg2_preproc/sub-{sub:02d}/vdvae/'
    performances_save_dir = f'results/thingseeg2_preproc/sub-{sub:02d}/performances_vdvae.npy'



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
