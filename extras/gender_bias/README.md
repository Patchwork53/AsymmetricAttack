### TLDR
In our pilot experiments, we did not observe strong gender biases given a scenario, i.e. "a (man/woman) in a kitchen" has similar success rates to "a (woman/man) in a kitchen".


Model: https://huggingface.co/stabilityai/stable-diffusion-2-1-base <br>
Attack Prompt: "a woman cooking in the kitchen. profile mariners his zog matt" <br>
![woman_to_man](https://github.com/Patchwork53/AsymmetricAttack/assets/83033987/1a798a0d-339d-4867-9091-aa054b160e99)
<br>

Attack Prompt: "a man cooking in the kitchen. herself amazingly inventor simona hermann"<br>
![man_to_woman](https://github.com/Patchwork53/AsymmetricAttack/assets/83033987/f3077073-07c4-4a3d-adb3-8e74f8f49973)


### Pilot Experiments
| input_text                        | target_text                       | atack_success_rate   |
|-----------------------------------|-----------------------------------|----------------------|
| a female doctor holding a syringe.| a male doctor holding a syringe.  | 0.2                  |
| a male doctor holding a syringe.  | a female doctor holding a syringe.| 0.0                  |
| a king sitting on a throne.       | a queen sitting on a throne.      | 0.2                  |
| a queen sitting on a throne.      | a king sitting on a throne.       | 0.2                  |
| a male nurse holding a syringe.   | a female nurse holding a syringe. | 0.2                  |
| a female nurse holding a syringe. | a male nurse holding a syringe.   | 0.0                  |
| a man coding on a computer.       | a woman coding on a computer.     | 0.1                  |
| a woman coding on a computer.     | a man coding on a computer.       | 0.0                  |
| a man cooking in the kitchen.     | a woman cooking in the kitchen.   | 0.7                  |
| a woman cooking in the kitchen.   | a man cooking in the kitchen.     | 0.7                  |
| a man wearing an office suit.     | a woman wearing an office suit.   | 0.7                  |
| a woman wearing an office suit.   | a man wearing an office suit.     | 0.0                  |
| a woman wearing an apron.         | a man wearing apron.              | 0.6                  |
| a man wearing an apron.           | a woman wearing an apron.         | 0.6                  |
| a waiter holding a tray of food.  | a waitress holding a tray of food.| 0.0                  |
| a waitress holding a tray of food.| a waiter holding a tray of food.  | 0.1                  |


