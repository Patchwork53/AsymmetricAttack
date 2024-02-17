import textwrap
from io import BytesIO

import requests
import torch
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
from tqdm import tqdm
import os
import json
import numpy as np
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str)
parser.add_argument("--num_prompts", type=int, default=10)
parser.add_argument("--num_images_per_prompt", type=int, default=5)
args = parser.parse_args()

# MODEL = "liuhaotian/llava-v1.5-13b"
MODEL = "liuhaotian/llava-v1.5-7b"
CONV_MODE = "llava_v0"
FOLDER = args.folder+"/"
NUM_PROMPTS = args.num_prompts
NUM_IMAGES_PER_PROMPT = args.num_images_per_prompt
NUM_IMAGES = NUM_PROMPTS*NUM_IMAGES_PER_PROMPT

model_name = get_model_name_from_path(MODEL)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL, model_base=None, model_name=model_name, load_4bit=True
)

def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def process_image(image):
    args = {"image_aspect_ratio": "pad"}
    image_tensor = process_images([image], image_processor, args)
    return image_tensor.to(model.device, dtype=torch.float16)


def create_prompt(prompt: str):
    conv = conv_templates[CONV_MODE].copy()
    roles = conv.roles
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)
    return conv.get_prompt(), conv

def ask_image(image: Image, prompt: str):
    image_tensor = process_image(image)
    prompt, conv = create_prompt(prompt)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria(
        keywords=[stop_str], tokenizer=tokenizer, input_ids=input_ids
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.01,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    return tokenizer.decode(
        output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
    ).strip()




all_prompts = []

THRESHOLD = 3
matches = 0
total = 0

for folder_name in tqdm(os.listdir(FOLDER)):
    folder = FOLDER + folder_name

    with open(folder + "/prompts.json", "r") as f:
        prompt = json.load(f)

  
    answers_wrt_input_text = []
    answers_wrt_target_text = []

    images = []
    for i in range(NUM_IMAGES):
        images.append(Image.open(folder + "/" + str(i) + ".jpg").convert("RGB"))

    template = "Does the image match the caption \"{}\"? Yes or No?"
    
    against_input  =  template.format(prompt["input_text"])
    against_target =  template.format(prompt["target_text"])

    scores = []

    for i in range(NUM_IMAGES):
        
        text_outputs1 = ask_image(images[i], against_input)
        text_outputs2 = ask_image(images[i], against_target)
        
        score = 0
        if text_outputs1 == "Yes":
            score -= 1
        if text_outputs2 == "Yes":
            score += 1
        if text_outputs1 not in ["Yes", "No"]:
            print("ERROR", i , text_outputs1)
        if text_outputs2 not in ["Yes", "No"]:
            print("ERROR", i , text_outputs2)
            
        scores.append(score)



    scores = np.array(scores)
    scores = scores.reshape(NUM_PROMPTS, NUM_IMAGES_PER_PROMPT)
    temp_list = scores.tolist()

    prompt["llava_scores"] = [[int(x) for x in y] for y in temp_list]



    success = 0
    for blip_score in scores:
        if blip_score.tolist().count(1) >= THRESHOLD:
            success += 1

    prompt["success"] = success
    
    print(prompt)

    all_prompts.append(prompt)
    
    

for i,p in enumerate(all_prompts):  
    with open( str(i)+".json", "w") as f:
        json.dump(p, f, indent=4)