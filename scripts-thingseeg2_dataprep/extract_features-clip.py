import requests
import torch
from PIL import Image
from io import BytesIO
import numpy as np
from tqdm import tqdm
import os

from diffusers import StableUnCLIPImg2ImgPipeline

pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
)
pipe = pipe.to("cuda")

# Load the test_images NumPy array
images = np.load("data/thingseeg2_metadata/test_images.npy", mmap_mode='r')
# Convert each image to a PIL image
images = [Image.fromarray(image).convert("RGB") for image in images]

embeddings = np.zeros((len(images), 1024))
device = pipe._execution_device
noise_level = torch.tensor([0], device=device)
torch.manual_seed(0)
for i_image, image in tqdm(enumerate(images), total=len(images)):
    embedding = pipe._encode_image(image, device=device, batch_size=1, num_images_per_prompt=1, do_classifier_free_guidance=True,noise_level=noise_level,generator=None, image_embeds = None)
    embedding = embedding[1]
    embeddings[i_image] = embedding.detach().cpu().numpy()[:1024]

os.makedirs('cache/thingseeg2_extracted_embeddings', exist_ok=True)
np.save('cache/thingseeg2_extracted_embeddings/test_clip.npy', embeddings)

# Load the train_images NumPy array
images = np.load("data/thingseeg2_metadata/train_images.npy", mmap_mode='r')
# Convert each image to a PIL image
images = [Image.fromarray(image).convert("RGB") for image in images]

embeddings = np.zeros((len(images), 1024))
device = pipe._execution_device
noise_level = torch.tensor([0], device=device)
torch.manual_seed(0)
for i_image, image in tqdm(enumerate(images), total=len(images)):
    embedding = pipe._encode_image(image, device=device, batch_size=1, num_images_per_prompt=1, do_classifier_free_guidance=True,noise_level=noise_level,generator=None, image_embeds = None)
    embedding = embedding[1]
    embeddings[i_image] = embedding.detach().cpu().numpy()[:1024]

os.makedirs('cache/thingseeg2_extracted_embeddings', exist_ok=True)
np.save('cache/thingseeg2_extracted_embeddings/train_clip.npy', embeddings)