For Text-to-Image Generation models, some prompts are harder than others. To evaluate the relative difficulty of prompts, we generate a large number (64) of images per prompt and use InstructBLIP to check if the Image generation was successful, i.e. the image matched the prompt.

Usage:
```
python generate_images-without_attack.py --csv_file prompts.csv --save_folder BASE
python blip_eval_images_wo_attack.py --input_folder BASE
```

The csv_file needs to have a column "input_text". The code snippet will create a success_rate.csv file in the working directory.