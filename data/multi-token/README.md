We carried out all attacks with [SD 2.1 BASE](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) and generated images with generation seed 27.

HQ-Pairs.zip and COCO-Pairs.zip contains the generated attack prompts, their success_rate according to InstructBLIP and other metrics.

For example:

```
{
    "input_text": "a bear is sitting in a green field",
    "target_text": "a woman is sitting in a green field",
    "num_tokens": 5,
    "adv_prompts": [
        "a bear is sitting in a green field bus \u2122 woman beagle examining",
        "a bear is sitting in a green field hornet woman pruning olive overlooking",
        "a bear is sitting in a green field woman collab observing roller process",
        "a bear is sitting in a green field woman sandro prepping scratches expression",
        "a bear is sitting in a green field concentrated woman pottery finisher woman",
        "a bear is sitting in a green field wife holds complexion examining sonja",
        "a bear is sitting in a green field thoroughbred woman artisan contemplating scrolling",
        "a bear is sitting in a green field amongst quantitative mages angelic women",
        "a bear is sitting in a green field tortoise congestion supervisor woman thinks",
        "a bear is sitting in a green field productions peaceful woman examining spinach"
    ],
    "blip_scores": [   # for each of the 10 adv_prompts, we generate 5 images.
        [
            1, # this image matched the target
            0, # this image matched neither the input or the target
            0,
            0,
            0
        ],
        [
            0,
            0,
            0,
            1,
            -1 # this image matched the input. (The adv suffix had no effect)
        ],
        [    #since a majority of the images matched the target, we consider this attack a success
            1,
            1,
            0,
            0,
            1
        ],
        [
            1,
            1,
            -1,
            1,
            -1
        ],
        [
            0,
            1,
            1,
            1,
            1
        ],
        [
            0,
            0,
            0,
            1,
            0
        ],
        [
            -1,
            -1,
            -1,
            -1,
            -1
        ],
        [
            -1,
            0,
            1,
            1,
            -1
        ],
        [
            1,
            1,
            -1,
            1,
            -1
        ],
        [
            1,
            1,
            1,
            1,
            1
        ]
    ]
}

```