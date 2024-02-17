Official implementation of [Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks](https://arxiv.org/abs/2312.14440)

# Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks
## Abstract
The widespread use of Text-to-Image (T2I) models in content generation requires careful examination of their safety, including their robustness to adversarial attacks. Despite extensive research into this, the reasons for their effectiveness are underexplored. This paper presents an empirical study on adversarial attacks against T2I models, focusing on analyzing factors associated with attack success rates (ASRs). We introduce a new attack objective - entity swapping using adversarial suffixes and two gradient-based attack algorithms. Human and automatic evaluations reveal the asymmetric nature of ASRs on entity swap: for example, it is easier to replace “human” with “robot” in the prompt “a human dancing in the rain.” with an adversarial suffix, but is significantly harder in reverse. We further propose probing metrics to establish indicative signals from the model’s beliefs to the adversarial ASR. We identify conditions resulting in a 60% success probability for adversarial attacks and others where this
likelihood drops below 5%.

![Capture](https://github.com/Patchwork53/AsymmetricAttack/assets/83033987/0aad810c-0ba2-44d2-9abe-56484923351b)


## Checklist:
- [x] Experimental Results (csv)
- [x] Attack Generation Code
- [x] Automated Evaluation Code
- [x] Base Success Rate Code
- [x] Generated Adversarial Attacks (csv)
- [x] Code to create COCO-Pairs
- [x] New results on COCO-Pairs

## Cite as:
```
@article{shahgir2023asymmetric,
  title={Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks},
  author={Shahgir, Haz Sameen and Kong, Xianghao and Steeg, Greg Ver and Dong, Yue},
  journal={arXiv preprint arXiv:2312.14440},
  year={2023}
}
```
