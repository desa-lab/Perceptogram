# %%
import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
import numpy as np
from tqdm import tqdm

# %%
# setup device to use
device = torch.device("cuda:6") if torch.cuda.is_available() else "cpu"

# %%
# we associate a model with its preprocessors to make it easier for inference.
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
# )

# Other available models:
#
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
)
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
# )
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
# )
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device
# )

vis_processors.keys()

# %% [markdown]
# ## train set

# %%
images = np.load('cache/thingsmeg/processed_data/BIGMEG1/test_images1b_sub-BIGMEG1.npy', mmap_mode='r')
captions = []
for i in tqdm(range(len(images)), total=len(images), desc="test captions"):
    image_pil = Image.fromarray(images[i].astype(np.uint8))
    image = vis_processors["eval"](image_pil).unsqueeze(0).to(device)
    captions.append(model.generate({"image": image})[0])
np.save('cache/thingsmeg/processed_data/BIGMEG1/test_captions1b_sub-BIGMEG1.npy', captions)

# %% [markdown]
# ## test set

# %%
images = np.load('cache/thingsmeg/processed_data/BIGMEG1/train_images1b_sub-BIGMEG1.npy', mmap_mode='r')
captions = []
for i in tqdm(range(len(images)), total=len(images), desc="train captions"):
    image_pil = Image.fromarray(images[i].astype(np.uint8))
    image = vis_processors["eval"](image_pil).unsqueeze(0).to(device)
    captions.append(model.generate({"image": image})[0])
np.save('cache/thingsmeg/processed_data/BIGMEG1/train_captions1b_sub-BIGMEG1.npy', captions)


