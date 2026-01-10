# Real-Time Scene Labeling for Autonomous Vehicles

BiSeNetV2-based semantic segmentation for real-time scene understanding in autonomous driving scenarios, trained on the Cityscapes dataset. The model segments road scenes into 19 semantic classes including roads, vehicles, pedestrians, and infrastructure.

cityscapes : leftImg8bit & gtFine

### Key Features
- Real-time Performance: 12.8MB model optimized for speed
- 19 Semantic Classes: Comprehensive scene understanding
- Optimized Pretrained Model: Ready-to-use weights for inference
- MPS/CUDA Support: GPU acceleration on macOS and Linux
- Easy Integration: Simple Python API for inference

## Quick Start

### Installation
```bash
# Install dependencies
uv sync
```

### Run Inference

**Real-time Webcam:**
I used my iPhone as a webcam via Camo, but you can also use the built-in webcam.

```bash
# Use built-in webcam
uv run python main.py --mode webcam

# Use iPhone as webcam (with Camo or Continuity Camera)
uv run python main.py --mode webcam --camera 2
```


## Getting Pretrained Model

1. Visit: https://www.kaggle.com/code/agampy/adversarial-patch-baseline/notebook
2. Download the BiSeNetV2 pretrained model
3. Place in: `pretrained_models/bisenetv2_cityscapes.pth`


## References

- Kaggle Notebook: [Adversarial Patch Baseline](https://www.kaggle.com/code/agampy/adversarial-patch-baseline/notebook)
- BiSeNetV2 Paper: "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
- Cityscapes Dataset: https://www.cityscapes-dataset.com/


## License

See the original Kaggle notebook for licensing information.
