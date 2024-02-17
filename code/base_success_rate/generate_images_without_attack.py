import os
import json
import torch
from diffusers import StableDiffusionPipeline
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline
from torch import autocast
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1-base")
parser.add_argument("--csv_file", type=str)
parser.add_argument("--save_folder", type=str)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--num_image_per_batch", type=int, default=16)

args = parser.parse_args()

SAVE_FOLDER = args.save_folder
NUM_IMAGE_PER_BATCH = args.num_image_per_batch
BATCH = args.batch


df = pd.read_csv(args.csv_file)
sentences = df["input_text"].tolist()

# model_id = "CompVis/stable-diffusion-v1-4"
model_id = args.model_id

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None)

pipe = pipe.to(device)
pipe.enable_attention_slicing()


def image_grid(imgs, cols):
    rows = len(imgs) // cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


torch.cuda.empty_cache()
for prompt in sentences:
    path = os.path.join(SAVE_FOLDER, prompt.replace(" ", "_").replace(".", ""))
    if not os.path.exists(path):
        os.makedirs(path)
    prompt_dict = {"prompt": prompt}

    with open(os.path.join(path, "prompt.json"), "w") as f:
        json.dump(prompt_dict, f)

    if len(os.listdir(path)) >= NUM_IMAGE_PER_BATCH*BATCH+2:
        print(f"Skipping Folder {path}")
        continue
    else:
        print(f"Folder {path} already exists but has {len(os.listdir(path))} images")

    images = []
    for b in range(BATCH):
        generator = torch.Generator("cuda").manual_seed(27+b)
        with autocast('cuda'):
            images += pipe([prompt]*NUM_IMAGE_PER_BATCH, generator=generator, num_inference_steps=30).images
        torch.cuda.empty_cache()

    grid = image_grid(images, 8)
    
    for image_idx, image in enumerate(images):
        image.save(os.path.join(path, f"{image_idx}.png"))
    

    
    grid.save(os.path.join(path, "grid.png"))


