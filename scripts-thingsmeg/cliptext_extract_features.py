import sys
sys.path.append('versatile_diffusion')
import os
import numpy as np

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
import torchvision.transforms as T

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.clip = net.clip.to(device)
   
train_caps = np.load('cache/thingsmeg/processed_data/BIGMEG1/train_captions1b_sub-BIGMEG1.npy', mmap_mode='r') 
test_caps = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_captions1b_sub-BIGMEG1.npy', mmap_mode='r')  
print(train_caps.shape, test_caps.shape)

num_embed, num_features, num_test, num_train = 77, 768, len(test_caps), len(train_caps)

train_clip = np.zeros((num_train,num_embed, num_features))
test_clip = np.zeros((num_test,num_embed, num_features))
with torch.no_grad():
    for i,annots in enumerate(test_caps):
        # cin = list(annots[annots!=''])
        cin = [annots]
        # print(i)
        print(i, cin)
        c = net.clip_encode_text(cin)
        test_clip[i] = c.to('cpu').numpy().mean(0)
    
    np.save('cache/thingsmeg/extracted_embeddings/BIGMEG1/test_cliptext1b_sub-BIGMEG1.npy',test_clip)
        
    for i,annots in enumerate(train_caps):
        # cin = list(annots[annots!=''])
        cin = [annots]
        # print(i)
        print(i, cin)
        c = net.clip_encode_text(cin)
        train_clip[i] = c.to('cpu').numpy().mean(0)
    np.save('cache/thingsmeg/extracted_embeddings/BIGMEG1/train_cliptext1b_sub-BIGMEG1.npy',train_clip)


