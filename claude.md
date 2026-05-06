# geo-classifier-quebec

Trains a region classifier for Quebec from street-level images. Given a street-level photo taken anywhere in Quebec, the model predicts which administrative region or geographic zone it was taken in.

The trained model is exported to ONNX and deployed on Triton Inference Server alongside other models (see cv-inference-triton repo).

## Project goal

Build a complete ML pipeline from scratch: data collection, cleaning, training, evaluation, and export. No existing dataset — images are collected via Mapillary API (free) and Google Street View Static API (~$20 budget).

## Hardware

- **Training:** Google Colab T4 (16GB VRAM)
- **Inference:** GTX 1060 3GB — model must be small enough to run inference comfortably within 3GB VRAM alongside other Triton models

## Stack

- Python, PyTorch, torchvision
- Mapillary API (primary image source, free)
- Google Street View Static API (gap-filling, paid ~$7/1000 images)
- osmnx (road network coordinate sampling)
- ONNX export for Triton deployment

## Project structure

```
data/           # collection and preprocessing scripts
train/          # training pipeline
eval/           # evaluation and error analysis
export/         # ONNX export for Triton deployment
```

## Data sources

- **Mapillary:** primary source, good coverage along the St. Lawrence corridor (Montreal → Quebec City → Ottawa)
- **Street View Static API:** used to fill gaps in regions with sparse Mapillary coverage
- Coverage was assessed per region before committing to scope — regions with insufficient images were excluded

## Key decisions

- Fine-tune a pretrained lightweight CNN (EfficientNet-B0 or MobileNetV3) — keeps inference well within the 3GB VRAM constraint
- Scope limited to regions with sufficient street-level image coverage — honesty about geographic gaps is part of the evaluation
