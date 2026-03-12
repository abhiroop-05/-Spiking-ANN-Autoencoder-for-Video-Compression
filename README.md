# HySNN — Hybrid Spiking Neural Network for Video Reconstruction

A lightweight, **0.536M-parameter** Spiking Neural Network (SNN) autoencoder that reconstructs video frames at high fidelity. Designed and trained on Kaggle (GPU) with a deployment target of **Raspberry Pi 4** via ONNX quantization.

---

## Overview

HySNN implements architecture that blends conventional and spiking neural network components to achieve an efficient balance between reconstruction quality and model size. The encoder uses Leaky Integrate-and-Fire (LIF) neurons with surrogate gradients for biologically-inspired temporal processing, while the decoder employs a symmetric upsampling path with skip connections.

Key highlights:

- **Test PSNR: 35.87 dB avg** over 240 frames on unseen video
- **Test SSIM: 0.9977** (near-perfect structural fidelity)
- **Model size: 2.07 MB** (PyTorch) / **1.84 MB** (INT8 ONNX)
- **GPU throughput: 31.3 FPS** @ 256×256 resolution
- **RasberryPi4 throughput: ~2.2 FPS** (FP32 ONNX, single-thread)

---

## Architecture

```
Input (3×256×256)
      │
  [Stem Conv + BN + ReLU]
      │
  ┌───▼──────────────────────────────────┐
  │  Encoder (Strategy3Encoder)          │
  │  DownBlock × 5  (32→48→64→48→32→24) │
  │  Bottleneck (16 ch)                  │
  │  DownsampledSkipConnections × 5      │
  └───────────────────────────────────────┘
      │  enc + skip_dense + skip_compact
  ┌───▼──────────────────────────────────┐
  │  Decoder (Strategy3Decoder)          │
  │  UpBlock × 5  (16→24→32→48→64→48)  │
  │  Refinement head → sigmoid           │
  └───────────────────────────────────────┘
      │
  Output (3×256×256)
```

### Key Modules

| Module | Description |
|---|---|
| `LIFNeuron` | Leaky Integrate-and-Fire neuron with soft reset and surrogate gradient |
| `SurrogateGradient` | Custom autograd function enabling backprop through spikes |
| `DepthwiseSeparableConv` | Efficient conv block with BN; falls back to standard conv for narrow channels |
| `ChannelAttention` | Squeeze-and-excitation style channel recalibration |
| `SpatialAttention` | Spatial feature refinement via avg/max pooling |
| `EfficientResidualBlock` | Residual block with dual spiking/ReLU activations and channel attention |
| `DownsampledSkipConnection` | Compressed skip connection with gated refinement |
| `Strategy3Autoencoder` | Top-level model combining encoder + decoder with 2 time steps |

---

## Results

### Training (rasp4.ipynb)

Trained for 30 epochs with early stopping (patience = 7). Best model saved at **epoch 20**.

| Metric | Value |
|---|---|
| Parameters | 0.536 M |
| Model size (PTH) | 2.07 MB |
| Test PSNR | **40.08 dB** |
| Test SSIM | **0.9986** |
| GPU Latency | 31.91 ms/frame |
| GPU Throughput | 31.3 FPS |

Dataset split: 210 train / 45 val / 45 test clips.

### Inference on Video (mycodetest.ipynb)

Evaluated on a 240-frame video at 30 FPS using the trained `best_model.pth`.

| Metric | Value |
|---|---|
| Avg PSNR | 35.87 dB |
| Avg SSIM | 0.9977 |
| Min PSNR | 35.44 dB (frame 97) |
| Max PSNR | 36.30 dB (frame 177) |
| Avg GPU Latency | 53.85 ms/frame |
| GPU Throughput | ~18.6 FPS |

### ONNX Export & Quantization

| Format | Size | CPU Throughput |
|---|---|---|
| FP32 ONNX | 2.57 MB | 2.2 FPS |
| INT8 ONNX | 1.84 MB | 0.3 FPS |

> **Note:** INT8 dynamic quantization currently degrades CPU throughput due to dequantization overhead. FP32 ONNX is recommended for Raspberry Pi 4 deployment.

---

## Repository Structure

```
.
├── rasp4.ipynb          # Training, evaluation, and ONNX export notebook
├── mycodetest.ipynb     # Inference + per-frame metric evaluation notebook
├── strategy3_256x256/
│   ├── best_model.pth   # Best PyTorch checkpoint
│   ├── model_fp32.onnx  # FP32 ONNX export
│   └── model_int8.onnx  # INT8 quantized ONNX export
└── test_output/
    ├── reconstructed_frames/   # Per-frame comparison PNGs
    ├── comparison_video.mp4    # Side-by-side original vs. reconstructed
    └── metrics_chart.png       # Per-frame PSNR / SSIM plots
```

---

## Getting Started

### Requirements

```bash
pip install torch torchvision opencv-python scikit-image matplotlib
pip install onnx onnxruntime
```

### Training

Open and run `rasp4.ipynb` on a GPU environment (Kaggle recommended). Configure data paths at the top of the notebook.

### Inference

Open `mycodetest.ipynb` and set the three config variables:

```python
VIDEO_PATH   = "path/to/your/video.mp4"
WEIGHTS_PATH = "path/to/best_model.pth"
OUTPUT_DIR   = "test_output"
```

Then run all cells. Outputs include per-frame PNG comparisons, a side-by-side MP4, and a metrics chart.

### Running ONNX on Raspberry Pi 4

```bash
pip install onnxruntime
```

Load the FP32 model:

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model_fp32.onnx", providers=["CPUExecutionProvider"])
inp  = np.random.randn(1, 3, 256, 256).astype(np.float32)  # replace with real frame
out  = sess.run(None, {sess.get_inputs()[0].name: inp})
```

---

## Hardware Targets

| Environment | Backend | Resolution | Throughput |
|---|---|---|---|
| Kaggle / Colab (GPU) | PyTorch (CUDA) | 256×256 | ~31 FPS |
| Desktop CPU | FP32 ONNX | 256×256 | ~2.2 FPS |
| Raspberry Pi 4 | FP32 ONNX | 256×256 | 1-2 FPS |

---

## License

This project is released for research and educational purposes.
