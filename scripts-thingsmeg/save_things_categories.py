import pandas as pd
import numpy as np

# df = pd.read_csv('THINGS-images/things_concepts.tsv',sep = '\t') 
# print(df.head())

## get text for each image

image_concept_index = pd.read_csv('THINGS-images/Metadata/Concept-specific/image_concept_index.csv', header=None).to_numpy()[:,0] - 1
print(image_concept_index.shape)
print(image_concept_index[:20])

words = pd.read_csv('THINGS-images/Metadata/Concept-specific/words.csv', header=None).to_numpy()[:,0]
print(words.shape)
print(words[:20])

text_np = np.array([words[id] for id in image_concept_index])
print(text_np[:20])
if not os.path.exists('data'):
    os.makedirs('data')
np.save('data/things_text_labels.npy', text_np)


# ## extract clip text
# import sys
# sys.path.append('versatile_diffusion')
# import torch
# from lib.cfg_helper import model_cfg_bank
# from lib.model_zoo import get_model
# from torch.utils.data import DataLoader, Dataset

# from lib.model_zoo.vd import VD
# from lib.cfg_holder import cfg_unique_holder as cfguh
# from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
# import torchvision.transforms as T
# from tqdm import tqdm

# cfgm_name = 'vd_noema'
# pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
# cfgm = model_cfg_bank()(cfgm_name)
# net = get_model()(cfgm)
# sd = torch.load(pth, map_location='cpu')
# net.load_state_dict(sd, strict=False)    

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net.clip = net.clip.to(device)

# num_embed, num_features, num_texts = 77, 768, len(text_np)
# clip = np.zeros((num_texts,num_embed,num_features))

# with torch.no_grad():
#     for i,cin in tqdm(enumerate(text_np), total=len(text_np)):
#         cin = [cin]
#         # print(i)
#         #ctemp = cin*2 - 1
#         c = net.clip_encode_text(cin)
#         clip[i] = c[0].cpu().numpy().mean(0)
    
#     np.save('data/extracted_features/things_cliptext.npy',clip)
