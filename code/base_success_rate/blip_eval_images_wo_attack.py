from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import json
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="Salesforce/instructblip-flan-t5-xl")
parser.add_argument("--input_folder", type=str)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--num_image_per_batch", type=int, default=16)

args = parser.parse_args()

FOLDER = args.input_folder
NUM_IMAGE_PER_BATCH = args.num_image_per_batch
BATCH = args.batch
NUM_IMAGES = BATCH * NUM_IMAGE_PER_BATCH

device = "cuda" if torch.cuda.is_available() else "cpu"
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl").to(device)
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

model.eval()
torch.cuda.empty_cache()



all_inputs = []
all_succ_rate = []

for folder_name in os.listdir(FOLDER):
    folder = FOLDER + folder_name

    with open(folder + "/prompt.json", "r") as f:
        prompt = json.load(f)

    
    template = "Does the image match the caption \"{}\"? Yes or No?"

    images = [Image.open(folder + "/" + x) for x in os.listdir(folder) if x.endswith(".png") and not x.startswith("grid")]

    inputs = processor(images=images, text=[template.format(prompt["prompt"])]*NUM_IMAGES, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs)
    text_outputs1 =  processor.batch_decode(outputs, skip_special_tokens=True)


    scores = []
    for i in range(NUM_IMAGES):
        score = 0
        if text_outputs1[i] == "Yes":
            score += 1

        if text_outputs1[i] not in ["Yes", "No"]:
            print("ERROR", i , text_outputs1[i])
            
        scores.append(score)


    success_rate = sum(scores)/len(scores)
    prompt["success_rate"] = success_rate

    print(prompt)
    with open(folder + "/prompt2.json", "w") as f:
        json.dump(prompt, f, indent=4)

    all_inputs.append(prompt["prompt"])
    all_succ_rate.append(success_rate)
    

for_csv = {
    "prompt": all_inputs,
    "success_rate": all_succ_rate
}



df = pd.DataFrame(for_csv)
df.to_csv("success_rate.csv", index=False)