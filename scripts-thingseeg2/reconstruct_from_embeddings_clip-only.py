import requests
import torch
from PIL import Image
from io import BytesIO
import numpy as np
import os

from diffusers import StableUnCLIPImg2ImgPipeline

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)

pred_clip = np.load(f"cache/thingseeg2_preproc/predicted_embeddings/sub-{sub:02d}/regress_clip.npy", mmap_mode='r') # Load the embeddings
recon_dir = f"results/thingseeg2_preproc/sub-{sub:02d}/unclip/" # Directory to save the reconstructed images
os.makedirs(recon_dir, exist_ok=True)

#Start the StableUnCLIP Image variations pipeline
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
)
pipe = pipe.to("cuda")
device = pipe._execution_device
torch_ones = torch.ones(512, dtype=torch.float16, device=device)
torch_zeros = torch.zeros(512, dtype=torch.float16, device=device)
extra_portion = torch.cat([torch_ones, torch_zeros])

for i, embedding in enumerate(pred_clip):
    print(i)
    embedding = torch.tensor(embedding, device=device, dtype=torch.float16)
    embedding = torch.cat([embedding, extra_portion]).unsqueeze(0)
    negative_prompt_embeds = torch.zeros_like(embedding)
    embedding = torch.cat([negative_prompt_embeds, embedding])
    torch.manual_seed(0)
    image = pipe.decode(embedding).images[0]
    image.save(recon_dir + f"{i}.png")