from PIL import Image
import requests
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import json
import numpy as np
import os
import torch
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str)
parser.add_argument("--num_prompts", type=int, default=10)
parser.add_argument("--num_images_per_prompt", type=int, default=5)
args = parser.parse_args()

NUM_PROMPTS = args.num_prompts
NUM_IMAGES_PER_PROMPT = args.num_images_per_prompt
FOLDER = args.folder+"/"

device = "cuda" if torch.cuda.is_available() else "cpu"

# model_name = "Salesforce/instructblip-flan-t5-xl"
# model_name = "Salesforce/instructblip-flan-t5-xxl"
# model_name = "Salesforce/instructblip-vicuna-13b"
model_name = "Salesforce/instructblip-vicuna-7b"
model = InstructBlipForConditionalGeneration.from_pretrained(model_name).to(device)
processor = InstructBlipProcessor.from_pretrained(model_name)


template = "Does the image match the caption \"{}\"? Yes or No?"

NUM_IMAGES = NUM_PROMPTS*NUM_IMAGES_PER_PROMPT
THRESHOLD = 3
matches = 0
total = 0

for folder_name in os.listdir(FOLDER):
    folder = FOLDER + folder_name

    with open(folder + "/prompts.json", "r") as f:
        prompt = json.load(f)
  
    answers_wrt_input_text = []
    answers_wrt_target_text = []

    images = []
    for i in range(NUM_IMAGES):
        images.append(Image.open(folder + "/" + str(i) + ".jpg").convert("RGB"))

   

    inputs = processor(images=images, text=[template.format(prompt["input_text"])]*50, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs)
    text_outputs1 =  processor.batch_decode(outputs, skip_special_tokens=True)

    inputs = processor(images=images, text=[template.format(prompt["target_text"])]*50, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs)
    text_outputs2 =  processor.batch_decode(outputs, skip_special_tokens=True)

    scores = []

    for i in range(NUM_IMAGES):
        score = 0
        if text_outputs1[i] == "Yes":
            score -= 1
        if text_outputs2[i] == "Yes":
            score += 1
        if text_outputs1[i] not in ["Yes", "No"]:
            print("ERROR", i , text_outputs1[i])
        if text_outputs2[i] not in ["Yes", "No"]:
            print("ERROR", i , text_outputs2[i])
            
        scores.append(score)

    scores = np.array(scores)
    scores = scores.reshape(NUM_PROMPTS, NUM_IMAGES_PER_PROMPT)
    temp_list = scores.tolist()

    prompt["blip_scores"] = [[int(x) for x in y] for y in temp_list]

    success = 0
    for blip_score in scores:
        if blip_score.tolist().count(1) >= THRESHOLD:
            success += 1

    prompt["success"] = success
    

    with open(folder + "/prompts.json", "w") as f:
        json.dump(prompt, f, indent=4)



for_csv2 = {
    "input_text": [],
    "target_text": [],
    "success": []
}

for folder_name in os.listdir(FOLDER):
    folder_name = os.path.join(FOLDER, folder_name)
    prompt = None
    with open(os.path.join(folder_name, 'prompts.json')) as f:
        prompt = json.load(f)
    
    for_csv2["input_text"].append(prompt["input_text"])
    for_csv2["target_text"].append(prompt["target_text"])
    for_csv2["success"].append(prompt["success"])

import pandas as pd

df = pd.DataFrame(for_csv2)

df.to_csv("blip_result.csv", index=False)
