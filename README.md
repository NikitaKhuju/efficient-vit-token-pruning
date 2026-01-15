# Efficient ViT via Token Pruning

This repository implements **efficient token pruning and merging (ToMe) for Vision Transformers (ViTs)** to reduce computational cost during inference while maintaining competitive accuracy. Token pruning removes less important image patch tokens, while **ToMe (Token Merging)** merges similar tokens to further reduce computation and memory usage.

---

## Dataset

We use the **ImageNet-100** dataset, a subset of ImageNet with 100 classes:

- Hugging Face: [clane9/imagenet-100](https://huggingface.co/datasets/clane9/imagenet-100)

---

## Pre-trained Model

The project uses the **ViT-Base (ViT-B/16)** model pre-trained on ImageNet-1k:

- Hugging Face: [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)

---

## ToMe (Token Merging)

- Paper: [ToMe: Token Merging for Efficient Vision Transformers](https://arxiv.org/abs/2303.09452)
- GitHub: [https://github.com/facebookresearch/ToMe](https://github.com/facebookresearch/ToMe)

---

## Dependencies

- Python 3
- torch (PyTorch)
- torchvision
- timm (optional)
- datasets (Hugging Face)
- numpy
- pandas
- matplotlib / seaborn

---

## Reference

- Dosovitskiy et al., _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale_, ICLR 2021
- Dynamic token pruning: [DynamicViT](https://arxiv.org/abs/2103.14030)

---
