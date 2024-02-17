import os
import shutil
import json
import torch
from diffusers import StableDiffusionPipeline
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import argparse
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1-base")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--start_thresh", type=float, default=1)
parser.add_argument("--end_thresh", type=float, default=0.25)
parser.add_argument("--num_adv_tokens", type=int, default=5)
parser.add_argument("--num_generations", type=int, default=10)
parser.add_argument("--num_images", type=int, default=5)
parser.add_argument("--timesteps", type=int, default=100)
parser.add_argument("--topk", type=int, default=256)
parser.add_argument("--max_match", type=int, default=512)
parser.add_argument("--constrained", type=bool, default=False)
parser.add_argument("--skip_tried_suffixes", type=bool, default=True)
args = parser.parse_args()

model_id = args.model_id
device = args.device
START_THRESH = args.start_thresh
END_THRESH = args.end_thresh
NUM_ADV_TOKENS = args.num_adv_tokens
NUM_GENERATIONS = args.num_generations
NUM_IMAGES = args.num_images
T = args.timesteps
k = args.topk
B_MAX = args.max_match
CONSTRAINED = args.constrained
SKIP_TRIED_SUFFIXES  = args.skip_tried_suffixes



pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None)
pipe = pipe.to(device)
pipe.enable_attention_slicing()


generator = torch.Generator(device).manual_seed(27)
tokenizer = pipe.tokenizer
pipe.text_encoder = pipe.text_encoder.float()
model = pipe.text_encoder
tokenizer.pad_token = tokenizer.eos_token
MAX_LENGTH = tokenizer.model_max_length


cos = torch.nn.CosineSimilarity(dim=0, eps=1e-06)
cos_dim1 = torch.nn.CosineSimilarity(dim=1, eps=1e-06)


allowed_char_idxes = None

if CONSTRAINED:
    allowed_chars = get_allowed_characters()
    allowed_char_idxes = tokenizer(allowed_chars, return_tensors="pt")["input_ids"][:,1]


input_target_data = []

with open("data_for_attacks.jsonl", "r") as f:
    for line in f:
        _json = json.loads(line)
        input_target_data.append((_json["input_text"],_json["target_text"]))


all_images = []
for input_text, target_text in input_target_data:

    input_tokens = tokenizer(input_text, return_tensors="pt", padding="max_length",max_length=MAX_LENGTH, truncation=True)["input_ids"].to(device)
    target_tokens = tokenizer(target_text, return_tensors="pt", padding="max_length",max_length=MAX_LENGTH, truncation=True)["input_ids"].to(device)

    target = None
    with torch.no_grad():
        target = model(target_tokens)[0][0].view(-1)

    avoid = None
    with torch.no_grad():
        avoid = model(input_tokens)[0][0].view(-1)

    input_tokens = input_tokens[0]

    num_adv_tokens = NUM_ADV_TOKENS
    suff_start = len(tokenizer(input_text).input_ids)-1
    suff_end = suff_start + num_adv_tokens

    print("Input:", input_text)
    print("Target:", target_text)
    print("Cosine Similarity between Input and Target", cos(avoid,target).item())


    tried_adv_tokens = set()

    if allowed_char_idxes is not None:
        k = len(allowed_char_idxes)
    B = min(k*num_adv_tokens, B_MAX)
    all_adv_input_tokens = [gradient_greedy_search(T, k, B, model, input_tokens, target, 
                                                   avoid, suff_start, suff_end, 
                                                   allowed_char_idxes,
                                                   tried_adv_tokens,
                                                   START_THRESH, 
                                                   END_THRESH, 
                                                   SKIP_TRIED_SUFFIXES, 
                                                   tokenizer, 
                                                   MAX_LENGTH
                                                   ) for i in range(NUM_GENERATIONS)]

    print("Number of inputs tried: ",len(tried_adv_tokens))

    if allowed_char_idxes is not None:
        print("Percentage of Search Space Explored:", 100*len(tried_adv_tokens)/(len(allowed_char_idxes)**num_adv_tokens),"%")


    images = []
    adv_prompts = []
    for adv_input_tokens in all_adv_input_tokens:
        final_adv_text = tokenizer.decode(adv_input_tokens, clean_up_tokenization_spaces =True,skip_special_tokens=True)
        print(final_adv_text)
        adv_prompts.append(final_adv_text)
        prompt = [final_adv_text] * NUM_IMAGES
        with autocast('cuda'):
            images += pipe(prompt, generator=generator, num_inference_steps=30).images

    
    grid = image_grid(images, rows=len(all_adv_input_tokens), cols=NUM_IMAGES)
    all_images.extend(images)
    
    dir_path = input_text.replace(" ","_").replace(".","_")+"___"+target_text.replace(" ","_").replace(".","_")+"/"


    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory '{dir_path}' has been removed")
        except OSError as e:
            print(f"Error: {dir_path} : {e.strerror}")


    os.makedirs(dir_path)

    prompts_dict = {
        "input_text":input_text,
        "target_text":target_text,
        "num_tokens":NUM_ADV_TOKENS,
        "adv_prompts":adv_prompts
    }

    with open(dir_path+"prompts.json", 'w') as file:
        json.dump(prompts_dict, file, indent=4) 

    for i,image in enumerate(images):
        image.save(dir_path+str(i)+".jpg")

    grid.save(dir_path+"grid.jpg")



all_grid = image_grid(all_images, rows=len(input_target_data)*NUM_GENERATIONS, cols=NUM_IMAGES)
all_grid = all_grid.resize((all_grid.width//8,all_grid.height//8))

all_grid.save("all_grid.jpg")
