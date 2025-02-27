import requests
import torch
from PIL import Image
from io import BytesIO
import numpy as np
from tqdm import tqdm

from diffusers import StableUnCLIPImg2ImgPipeline
# device='cuda:0'
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
)
pipe = pipe.to('cuda')

# Load the test_images NumPy array
images = np.load("data/thingseeg2_metadata/test_images.npy", mmap_mode='r')

print('Resizing images')
# Initialize an empty list to store the resized images
resized_images = []
# resized_images = np.zeros((len(images), 768, 768, 3))
# Iterate over each image
for i_image, image in tqdm(enumerate(images), total=len(images)):
    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image)
    # Resize the image to 768x768
    resized_pil_image = pil_image.resize((768, 768), Image.LANCZOS)
    # Convert the PIL Image back to a NumPy array
    resized_image = np.array(resized_pil_image)
    # Append the resized image to the list
    resized_images.append(resized_image)
    # resized_images[i_image] = resized_image
# Convert the list of resized images back to a NumPy array
# images = np.array(resized_images)
images = resized_images

print('Encoding images')
embeddings = np.zeros((len(images), 36864))
device = pipe._execution_device
torch.manual_seed(0)
for i_image, image in tqdm(enumerate(images), total=len(images)):
    tensor_image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor_image = tensor_image.unsqueeze(0).to(device).half()
    vae_latent = pipe.vae.encode(tensor_image, return_dict=False)[0].mode()
    vae_latent_np = vae_latent.detach().cpu().numpy()
    vae_latent_np_flatten = vae_latent_np.flatten()
    embeddings[i_image] = vae_latent_np_flatten

np.save('cache/thingseeg2_extracted_embeddings/test_vae.npy', embeddings)

# Load the train_images NumPy array
images = np.load("data/thingseeg2_metadata/train_images.npy", mmap_mode='r')

print('Resizing images')
# Initialize an empty list to store the resized images
resized_images = []
# resized_images = np.zeros((len(images), 768, 768, 3))
# Iterate over each image
for i_image, image in tqdm(enumerate(images), total=len(images)):
    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image)
    # Resize the image to 768x768
    resized_pil_image = pil_image.resize((768, 768), Image.LANCZOS)
    # Convert the PIL Image back to a NumPy array
    resized_image = np.array(resized_pil_image)
    # Append the resized image to the list
    resized_images.append(resized_image)
    # resized_images[i_image] = resized_image
# Convert the list of resized images back to a NumPy array
# images = np.array(resized_images)
images = resized_images

print('Encoding images')
embeddings = np.zeros((len(images), 36864))
device = pipe._execution_device
torch.manual_seed(0)
for i_image, image in tqdm(enumerate(images), total=len(images)):
    tensor_image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor_image = tensor_image.unsqueeze(0).to(device).half()
    vae_latent = pipe.vae.encode(tensor_image, return_dict=False)[0].mode()
    vae_latent_np = vae_latent.detach().cpu().numpy()
    vae_latent_np_flatten = vae_latent_np.flatten()
    embeddings[i_image] = vae_latent_np_flatten

np.save('cache/thingseeg2_extracted_embeddings/train_vae.npy', embeddings)
