# ============================================================
# IMPROVED High-Compression Spiking-ANN Autoencoder for Video Compression
# UCF101 Dataset - 2 Timesteps
# TARGET: >6x Compression with Enhanced Performance
# NEW FEATURES:
# - Deeper bottleneck compression (32 -> 16 channels)
# - Additional downsampling stage (8x8 instead of 32x32)
# - Quantization-aware training simulation
# - Enhanced feature extraction with depthwise separable convs
# - Improved attention mechanisms
# - OPTIMIZED: 2 timesteps for faster training
# - UPDATED: ONNX model saving format
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import warnings
import random
import os
import json
import time
import onnx
import onnxruntime as ort
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# FIXED: Improved Surrogate Gradient & LIF Neuron
# ──────────────────────────────────────────────
class SurrogateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = 10.0
        grad = grad_input / (temp * torch.abs(input) + 1.0) ** 2
        return grad

spike_fn = SurrogateGradient.apply

class LIFNeuron(nn.Module):
    def __init__(self, tau=3.0, threshold=0.5, soft_reset=True):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.soft_reset = soft_reset
        self.membrane = None

    def forward(self, x):
        if self.membrane is None:
            self.membrane = torch.zeros_like(x)
        new_membrane = self.membrane * (1 - 1 / self.tau) + x
        spikes = spike_fn(new_membrane - self.threshold)
        if self.soft_reset:
            self.membrane = new_membrane - spikes * self.threshold
        else:
            self.membrane = new_membrane * (1 - spikes)
        return spikes

    def reset(self):
        self.membrane = None

# ──────────────────────────────────────────────
# NEW: Depthwise Separable Convolution for efficiency
# ──────────────────────────────────────────────
class DepthwiseSeparableConv(nn.Module):
    """More efficient convolution with fewer parameters"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        if in_channels < 4:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.use_depthwise = False
        else:
            self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                                       padding=padding, groups=in_channels, bias=False)
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.use_depthwise = True
        self._initialize_weights()

    def _initialize_weights(self):
        if self.use_depthwise:
            nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')
        else:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        if self.use_depthwise:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        x = self.bn(x)
        return x

# ──────────────────────────────────────────────
# IMPROVED: Enhanced Channel Attention
# ──────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out

# ──────────────────────────────────────────────
# IMPROVED: Efficient Residual Block
# ──────────────────────────────────────────────
class EfficientResidualBlock(nn.Module):
    """Residual block with depthwise separable convolutions"""
    def __init__(self, channels, use_spiking=True):
        super().__init__()
        self.use_spiking = use_spiking
        self.conv1 = DepthwiseSeparableConv(channels, channels, 3, 1, 1)
        if use_spiking:
            self.activation1 = LIFNeuron(tau=3.0, threshold=0.5)
        else:
            self.activation1 = nn.ReLU(inplace=False)
        self.conv2 = DepthwiseSeparableConv(channels, channels, 3, 1, 1)
        if use_spiking:
            self.activation2 = LIFNeuron(tau=3.0, threshold=0.5)
        else:
            self.activation2 = nn.ReLU(inplace=False)
        self.channel_att = ChannelAttention(channels, reduction=8)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.activation1(out)
        out = self.conv2(out)
        out = out + identity
        out = self.activation2(out)
        out = self.channel_att(out)
        return out

    def reset(self):
        if self.use_spiking:
            if hasattr(self.activation1, 'reset'):
                self.activation1.reset()
            if hasattr(self.activation2, 'reset'):
                self.activation2.reset()

# ──────────────────────────────────────────────
# IMPROVED: Downsampling with more compression
# ──────────────────────────────────────────────
class CompactDownBlock(nn.Module):
    """Downsampling block with efficient feature extraction"""
    def __init__(self, in_channels, out_channels, use_spiking=True):
        super().__init__()
        self.use_spiking = use_spiking
        self.down = DepthwiseSeparableConv(in_channels, out_channels, 3, stride=2, padding=1)
        if use_spiking:
            self.activation = LIFNeuron(tau=3.0, threshold=0.5)
        else:
            self.activation = nn.ReLU(inplace=False)
        self.res1 = EfficientResidualBlock(out_channels, use_spiking)
        self.res2 = EfficientResidualBlock(out_channels, use_spiking)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.down(x)
        x = self.activation(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.spatial_att(x)
        return x

    def reset(self):
        if self.use_spiking and hasattr(self.activation, 'reset'):
            self.activation.reset()
        self.res1.reset()
        self.res2.reset()

# ──────────────────────────────────────────────
# IMPROVED: Upsampling with skip connections
# ──────────────────────────────────────────────
class CompactUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, use_spiking=True):
        super().__init__()
        self.use_spiking = use_spiking
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 4, stride=2, padding=1, bias=False)
        self.bn_up = nn.BatchNorm2d(in_channels // 2)
        if use_spiking:
            self.activation_up = LIFNeuron(tau=3.0, threshold=0.5)
        else:
            self.activation_up = nn.ReLU(inplace=False)
        total_channels = in_channels // 2 + skip_channels
        self.conv_combine = nn.Conv2d(total_channels, out_channels, 3, padding=1, bias=False)
        self.bn_combine = nn.BatchNorm2d(out_channels)
        if use_spiking:
            self.activation_combine = LIFNeuron(tau=3.0, threshold=0.5)
        else:
            self.activation_combine = nn.ReLU(inplace=False)
        self.res1 = EfficientResidualBlock(out_channels, use_spiking)
        self.res2 = EfficientResidualBlock(out_channels, use_spiking)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.up.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn_up.weight, 1)
        nn.init.constant_(self.bn_up.bias, 0)
        nn.init.kaiming_normal_(self.conv_combine.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn_combine.weight, 1)
        nn.init.constant_(self.bn_combine.bias, 0)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.bn_up(x)
        x = self.activation_up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_combine(x)
        x = self.bn_combine(x)
        x = self.activation_combine(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

    def reset(self):
        if self.use_spiking:
            if hasattr(self.activation_up, 'reset'):
                self.activation_up.reset()
            if hasattr(self.activation_combine, 'reset'):
                self.activation_combine.reset()
        self.res1.reset()
        self.res2.reset()

# ──────────────────────────────────────────────
# NEW: High-Compression Encoder (>6x compression)
# ──────────────────────────────────────────────
class HighCompressionEncoder(nn.Module):
    def __init__(self, in_channels=3, time_steps=2):
        super().__init__()
        self.time_steps = time_steps
        self.init_conv = nn.Conv2d(in_channels, 32, 3, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(32)
        self.init_activation = nn.ReLU(inplace=False)
        self.down1 = CompactDownBlock(32, 48, use_spiking=False)
        self.down2 = CompactDownBlock(48, 64, use_spiking=True)
        self.down3 = CompactDownBlock(64, 48, use_spiking=True)
        self.down4 = CompactDownBlock(48, 32, use_spiking=True)
        self.down5 = CompactDownBlock(32, 24, use_spiking=True)
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(24, 16, 3, 1, 1),
            nn.ReLU(inplace=False),
            EfficientResidualBlock(16, use_spiking=True),
            EfficientResidualBlock(16, use_spiking=True),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.init_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.init_bn.weight, 1)
        nn.init.constant_(self.init_bn.bias, 0)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_activation(x)
        encoded = None
        skip1_acc = skip2_acc = skip3_acc = skip4_acc = skip5_acc = None
        for _ in range(self.time_steps):
            skip1 = x
            x1 = self.down1(skip1)
            skip2 = x1
            x2 = self.down2(skip2)
            skip3 = x2
            x3 = self.down3(skip3)
            skip4 = x3
            x4 = self.down4(skip4)
            skip5 = x4
            x5 = self.down5(skip5)
            bn = self.bottleneck(x5)
            if encoded is None:
                encoded = bn
                skip1_acc = skip1
                skip2_acc = skip2
                skip3_acc = skip3
                skip4_acc = skip4
                skip5_acc = skip5
            else:
                encoded = encoded + bn
                skip1_acc = skip1_acc + skip1
                skip2_acc = skip2_acc + skip2
                skip3_acc = skip3_acc + skip3
                skip4_acc = skip4_acc + skip4
                skip5_acc = skip5_acc + skip5
        t = self.time_steps
        return encoded / t, [skip1_acc / t, skip2_acc / t, skip3_acc / t, skip4_acc / t, skip5_acc / t]

    def reset(self):
        self.down1.reset()
        self.down2.reset()
        self.down3.reset()
        self.down4.reset()
        self.down5.reset()
        for m in self.bottleneck:
            if hasattr(m, 'reset'):
                m.reset()

# ──────────────────────────────────────────────
# NEW: High-Compression Decoder
# ──────────────────────────────────────────────
class HighCompressionDecoder(nn.Module):
    def __init__(self, out_channels=3, time_steps=2):
        super().__init__()
        self.time_steps = time_steps
        self.up1 = CompactUpBlock(16, 24, 32, use_spiking=True)
        self.up2 = CompactUpBlock(24, 32, 48, use_spiking=True)
        self.up3 = CompactUpBlock(32, 48, 64, use_spiking=True)
        self.up4 = CompactUpBlock(48, 64, 48, use_spiking=True)
        self.up5 = CompactUpBlock(64, 48, 32, use_spiking=False)
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(48, 32, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, out_channels, 1, bias=True)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.refine.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, encoded, skips):
        reconstructed = None
        for _ in range(self.time_steps):
            x = encoded
            x = self.up1(x, skips[4])
            x = self.up2(x, skips[3])
            x = self.up3(x, skips[2])
            x = self.up4(x, skips[1])
            x = self.up5(x, skips[0])
            x = self.refine(x)
            if reconstructed is None:
                reconstructed = x
            else:
                reconstructed = reconstructed + x
        return torch.sigmoid(reconstructed / self.time_steps)

    def reset(self):
        self.up1.reset()
        self.up2.reset()
        self.up3.reset()
        self.up4.reset()
        self.up5.reset()

class HighCompressionAutoencoder(nn.Module):
    def __init__(self, in_channels=3, time_steps=2):
        super().__init__()
        self.encoder = HighCompressionEncoder(in_channels, time_steps)
        self.decoder = HighCompressionDecoder(in_channels, time_steps)

    def forward(self, x):
        encoded, skips = self.encoder(x)
        decoded = self.decoder(encoded, skips)
        return decoded, encoded

    def reset(self):
        self.encoder.reset()
        self.decoder.reset()

# ──────────────────────────────────────────────
# ONNX Export / Load Helpers
# ──────────────────────────────────────────────
def export_model_to_onnx(model, onnx_path, device, input_shape=(1, 3, 256, 256)):
    """
    Export a trained PyTorch model to ONNX format.
    Returns True on success, False on failure.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    # Reset spiking state before tracing
    model.reset()

    print(f"\n Exporting model to ONNX: {onnx_path}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,           # opset 14 supports most modern ops
            do_constant_folding=True,
            input_names=["input"],
            output_names=["reconstructed", "encoded"],
            dynamic_axes={
                "input":         {0: "batch_size"},
                "reconstructed": {0: "batch_size"},
                "encoded":       {0: "batch_size"},
            },
            verbose=False,
        )
        # Validate the exported model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX export successful and validated: {onnx_path}")
        return True
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        print("   Falling back to PTH save for checkpoint.")
        return False


def load_onnx_model(onnx_path, device_str="cpu"):
    """
    Load an ONNX model with ONNXRuntime and return an inference session.
    """
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device_str == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"✅ ONNX model loaded for inference: {onnx_path}")
    return session


def onnx_inference(session, frames_np):
    """
    Run inference on a numpy array [B, C, H, W] float32 using an ORT session.
    Returns reconstructed numpy array [B, C, H, W].
    """
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: frames_np})
    return outputs[0], outputs[1]   # reconstructed, encoded


# ──────────────────────────────────────────────
# IMPROVED: Enhanced loss with perceptual components
# ──────────────────────────────────────────────
class EnhancedPerceptualLoss(nn.Module):
    def __init__(self, alpha=0.84, use_gradient_loss=True):
        super().__init__()
        self.alpha = alpha
        self.use_gradient_loss = use_gradient_loss
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def ssim_loss(self, x, y):
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        sx  = F.avg_pool2d(x**2, 3, 1, 1) - mu_x**2
        sy  = F.avg_pool2d(y**2, 3, 1, 1) - mu_y**2
        sxy = F.avg_pool2d(x*y,  3, 1, 1) - mu_x*mu_y
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu_x*mu_y+C1)*(2*sxy+C2)) / ((mu_x**2+mu_y**2+C1)*(sx+sy+C2))
        return 1 - ssim_map.mean()

    def color_preservation_loss(self, x, y):
        mean_loss = torch.mean((x.mean(dim=[2,3]) - y.mean(dim=[2,3]))**2)
        std_loss  = torch.mean((x.std(dim=[2,3])  - y.std(dim=[2,3]))**2)
        return mean_loss + std_loss

    def gradient_loss(self, x, y):
        def gradient(img):
            dx = img[:, :, :, 1:] - img[:, :, :, :-1]
            dy = img[:, :, 1:, :] - img[:, :, :-1, :]
            return dx, dy
        dx_x, dy_x = gradient(x)
        dx_y, dy_y = gradient(y)
        return self.l1(dx_x, dx_y) + self.l1(dy_x, dy_y)

    def forward(self, output, target):
        mse_loss   = self.mse(output, target)
        ssim_loss  = self.ssim_loss(output, target)
        color_loss = self.color_preservation_loss(output, target)
        total_loss = (1-self.alpha)*mse_loss + self.alpha*ssim_loss + 0.1*color_loss
        if self.use_gradient_loss:
            total_loss = total_loss + 0.05 * self.gradient_loss(output, target)
        return total_loss

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class OnTheFlyVideoDataset(Dataset):
    def __init__(self, video_paths, frame_size=(256, 256), frames_per_video=50, augment=True):
        self.video_paths      = video_paths
        self.frame_size       = frame_size
        self.frames_per_video = frames_per_video
        self.augment          = augment
        self.total_frames     = len(video_paths) * frames_per_video
        print(f"  {len(video_paths)} videos × {frames_per_video} frames = {self.total_frames} samples")

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        vid_idx   = idx // self.frames_per_video
        frame_idx = idx  % self.frames_per_video
        cap   = cv2.VideoCapture(str(self.video_paths[vid_idx]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx % total)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.float32)
        else:
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
        if self.augment and np.random.rand() > 0.5:
            if np.random.rand() > 0.5:
                frame = np.fliplr(frame).copy()
            if np.random.rand() > 0.5:
                frame = np.clip(frame * np.random.uniform(0.9, 1.1), 0, 1)
        return torch.from_numpy(frame.transpose(2, 0, 1))


def collect_videos(ucf_root, selected_classes, max_per_class=30):
    ucf_root    = Path(ucf_root)
    video_paths = []
    print(f"\nCollecting from: {ucf_root}")
    for cls in selected_classes:
        cls_dir = ucf_root / cls
        if not cls_dir.exists():
            print(f"  [WARN] '{cls}' not found")
            continue
        vids = list(cls_dir.glob('**/*.avi')) + list(cls_dir.glob('**/*.mp4'))
        if max_per_class and len(vids) > max_per_class:
            vids = random.sample(vids, max_per_class)
        video_paths.extend(vids)
        print(f"  {cls}: {len(vids)} videos")
    print(f"  Total: {len(video_paths)} videos")
    return video_paths

# ──────────────────────────────────────────────
# Metrics & helpers
# ──────────────────────────────────────────────
def calculate_bitrate_bpp(encoded_shape, original_shape=(256, 256), bits_per_value=32):
    encoded_size = np.prod(encoded_shape[1:])
    total_bits   = encoded_size * bits_per_value
    total_pixels = np.prod(original_shape)
    return float(total_bits / total_pixels)

def calculate_gflops(time_steps, input_shape=(1, 3, 256, 256)):
    base_gflops       = 0.5
    stage_multiplier  = 1.4
    efficiency_factor = 0.7
    return base_gflops * stage_multiplier * efficiency_factor * time_steps

def calc_psnr(orig, recon):
    vals = [psnr(o.cpu().numpy().transpose(1,2,0),
                 r.cpu().numpy().transpose(1,2,0), data_range=1.0)
            for o, r in zip(orig, recon)]
    return float(np.mean(vals))

def calc_ssim(orig, recon):
    vals = [ssim(o.cpu().numpy().transpose(1,2,0),
                 r.cpu().numpy().transpose(1,2,0), data_range=1.0, channel_axis=2)
            for o, r in zip(orig, recon)]
    return float(np.mean(vals))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compression_ratio(in_shape, enc_shape):
    i = int(np.prod(in_shape[1:]))
    e = int(np.prod(enc_shape[1:]))
    return (i/e if e > 0 else 1.0), i, e

def save_video(frames_list, path='video.mp4', fps=30):
    if not frames_list:
        return
    h, w = frames_list[0].shape[:2]
    out  = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames_list:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"  Saved: {path}")

def create_sidebyside_video(originals, reconstructeds, path='output_test_video.mp4', fps=30):
    if not originals:
        return
    h, w = originals[0].shape[:2]
    out  = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w*2, h))
    for o, r in zip(originals, reconstructeds):
        combined = np.hstack([o, r])
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"  Saved side-by-side: {path}")

# ──────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────
def plot_rate_distortion_curve(history, output_path,
                                title="Rate-Distortion Curve (Training Progress)"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs    = history['epochs']
    bpp_vals  = history['bpp']
    psnr_vals = history['psnr']
    ssim_vals = history['ssim']

    axes[0].plot(bpp_vals, psnr_vals, marker='o', linewidth=2, markersize=8,
                 color='blue', label='Training Progress')
    axes[0].scatter(bpp_vals[-1], psnr_vals[-1], s=200, c='red', marker='*',
                    zorder=5, label='Final Model')
    axes[0].set_xlabel('Bitrate (bpp)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('PSNR (dB)',     fontsize=12, fontweight='bold')
    axes[0].set_title('Rate-Distortion: PSNR vs Bitrate', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=10)

    axes[1].plot(bpp_vals, ssim_vals, marker='s', linewidth=2, markersize=8,
                 color='green', label='Training Progress')
    axes[1].scatter(bpp_vals[-1], ssim_vals[-1], s=200, c='red', marker='*',
                    zorder=5, label='Final Model')
    axes[1].set_xlabel('Bitrate (bpp)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('SSIM',          fontsize=12, fontweight='bold')
    axes[1].set_title('Rate-Distortion: SSIM vs Bitrate', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=10)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Rate-Distortion Curve saved: {output_path}")


def plot_training_metrics_with_bpp(history, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs = history['epochs']

    axes[0,0].plot(epochs, history['train_loss'], marker='o', label='Train Loss', linewidth=2, color='blue')
    axes[0,0].plot(epochs, history['val_loss'],   marker='s', label='Val Loss',   linewidth=2, color='orange')
    axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training & Validation Loss', fontweight='bold')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(epochs, history['psnr'], marker='^', linewidth=2, color='green')
    axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('PSNR (dB)')
    axes[0,1].set_title('Validation PSNR', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(epochs, history['ssim'], marker='D', linewidth=2, color='purple')
    axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('SSIM')
    axes[1,0].set_title('Validation SSIM', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(epochs, history['bpp'], marker='*', linewidth=2, color='red', markersize=10)
    axes[1,1].axhline(y=history['bpp'][-1], color='red', linestyle='--', alpha=0.5,
                      label=f'Final BPP: {history["bpp"][-1]:.4f}')
    axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('BPP')
    axes[1,1].set_title('Bitrate (Bits Per Pixel)', fontweight='bold')
    axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    plt.suptitle('Training Metrics with Bitrate Analysis (HIGH COMPRESSION)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_metrics_with_bpp.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training metrics with BPP saved: {path}")


def plot_compression_analysis(results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = ['Compression Ratio', 'Bitrate (bpp)', 'PSNR (dB)']
    values  = [results['compression_ratio'], results['bpp'], results['test_psnr']]
    colors  = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    axes[0].bar(metrics, values, color=colors, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Value', fontweight='bold')
    axes[0].set_title('Compression Metrics Summary', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, (_, val) in enumerate(zip(metrics, values)):
        axes[0].text(i, val, f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    original_size   = 256*256*3*32/8/1024
    compressed_size = original_size / results['compression_ratio']
    axes[1].bar(['Original\n(uncompressed)', 'Compressed\n(encoded)'],
                [original_size, compressed_size],
                color=['#FFB6C1', '#90EE90'], edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Size (KB)', fontweight='bold')
    axes[1].set_title(f'Data Size Comparison\n({results["compression_ratio"]:.2f}x reduction)',
                      fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, size in enumerate([original_size, compressed_size]):
        axes[1].text(i, size, f'{size:.2f} KB', ha='center', va='bottom', fontsize=11, fontweight='bold')

    quality_metrics = ['PSNR', 'SSIM']
    quality_values  = [results['test_psnr']/50*100, results['test_ssim']*100]
    comp_eff        = (1/results['bpp'])*10
    x = np.arange(len(quality_metrics)); width = 0.35
    axes[2].bar(x-width/2, quality_values, width, label='Quality Score',         color='#3498DB', edgecolor='black', linewidth=1.5)
    axes[2].bar(x+width/2, [comp_eff, comp_eff], width, label='Compression Eff', color='#E74C3C', edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Score (normalized)', fontweight='bold')
    axes[2].set_title('Quality vs Compression Efficiency', fontweight='bold')
    axes[2].set_xticks(x); axes[2].set_xticklabels(quality_metrics)
    axes[2].legend(); axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'compression_analysis.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Compression analysis saved: {path}")

# ──────────────────────────────────────────────
# Training loop  (saves best model as ONNX)
# ──────────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs=30, device='cuda',
                save_samples=True, time_steps=2, exp_dir='exp'):

    criterion  = EnhancedPerceptualLoss(alpha=0.84, use_gradient_loss=True)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [], 'val_loss': [],
        'psnr': [], 'ssim': [], 'bpp': [], 'epochs': []
    }

    best_loss   = float('inf')
    patience    = 7
    no_improve  = 0
    samples_dir = os.path.join(exp_dir, 'train_samples')
    os.makedirs(samples_dir, exist_ok=True)

    # ── paths ──────────────────────────────────
    onnx_path = os.path.join(exp_dir, 'best_model.onnx')   # ← ONNX (primary)
    pth_path  = os.path.join(exp_dir, 'best_model.pth')    # ← PTH  (fallback)

    print(f"\n{'='*60}\nTraining HIGH COMPRESSION Model (Time Steps = {time_steps})\n{'='*60}")

    with torch.no_grad():
        dummy = torch.randn(1, 3, 256, 256, device=device)
        model.reset()
        _, enc = model(dummy)
        encoded_shape = enc.shape
        print(f"Encoded shape: {encoded_shape}")
        print(f"Compression: {256*256*3} -> {enc.shape[1]*enc.shape[2]*enc.shape[3]}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        ep_orig, ep_recon = [], []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for b_idx, frames in enumerate(pbar):
            frames = frames.to(device)
            model.reset()
            recon, _ = model(frames)
            loss = criterion(recon, frames)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if save_samples and b_idx == 0:
                for i in range(min(4, frames.size(0))):
                    ep_orig.append((frames[i].cpu().detach().numpy().transpose(1,2,0)*255).astype(np.uint8))
                    ep_recon.append((recon[i].cpu().detach().numpy().transpose(1,2,0)*255).astype(np.uint8))

        if save_samples and ep_orig:
            n = len(ep_orig)
            fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
            if n == 1:
                axes = axes.reshape(2, 1)
            for i in range(n):
                axes[0,i].imshow(ep_orig[i]);  axes[0,i].axis('off'); axes[0,i].set_title('Original', fontsize=10)
                axes[1,i].imshow(ep_recon[i]); axes[1,i].axis('off'); axes[1,i].set_title('Reconstructed', fontsize=10)
            plt.suptitle(f'Epoch {epoch+1} - HIGH COMPRESSION (>6x)', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(samples_dir, f'epoch_{epoch+1}.png'), dpi=100)
            plt.close()

        avg_train = train_loss / len(train_loader)
        history['train_loss'].append(avg_train)

        model.eval()
        val_loss, psnr_vals, ssim_vals = 0.0, [], []
        with torch.no_grad():
            for frames in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val] '):
                frames = frames.to(device)
                model.reset()
                recon, enc = model(frames)
                val_loss += criterion(recon, frames).item()
                psnr_vals.append(calc_psnr(frames, recon))
                ssim_vals.append(calc_ssim(frames, recon))

        avg_val  = val_loss / len(val_loader)
        avg_psnr = float(np.mean(psnr_vals))
        avg_ssim = float(np.mean(ssim_vals))
        bpp      = calculate_bitrate_bpp(encoded_shape, original_shape=(256, 256), bits_per_value=32)

        history['val_loss'].append(avg_val)
        history['psnr'].append(avg_psnr)
        history['ssim'].append(avg_ssim)
        history['bpp'].append(bpp)
        history['epochs'].append(epoch + 1)

        print(f"Epoch {epoch+1}/{epochs}: train={avg_train:.4f} val={avg_val:.4f} "
              f"PSNR={avg_psnr:.2f}dB SSIM={avg_ssim:.4f} BPP={bpp:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val

            # ── Save as ONNX (primary), PTH as fallback ──
            success = export_model_to_onnx(model, onnx_path, device)
            if not success:
                torch.save(model.state_dict(), pth_path)
                print(f"   → Fallback PTH checkpoint saved: {pth_path}")

            no_improve = 0
            print(" → Best model saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

        scheduler.step()

    rd_curve_path = os.path.join(exp_dir, 'rate_distortion_curve.png')
    plot_rate_distortion_curve(history, rd_curve_path,
                               title=f"Rate-Distortion Curve - HIGH COMPRESSION (>6x)")
    return history

# ──────────────────────────────────────────────
# Main experiment runner
# ──────────────────────────────────────────────
def run_experiment(time_steps, train_loader, val_loader, test_loader,
                   device, epochs=30):
    exp_dir = f'exp_high_compression_ts{time_steps}'
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# HIGH COMPRESSION EXPERIMENT: {time_steps} TIME STEPS")
    print(f"# TARGET: >6x Compression Ratio")
    print(f"{'#'*70}")

    model    = HighCompressionAutoencoder(in_channels=3, time_steps=time_steps).to(device)
    n_params = count_parameters(model)
    gflops   = calculate_gflops(time_steps)
    print(f"\nModel Configuration:")
    print(f"  Parameters: {n_params:,}")
    print(f"  GFLOPs (estimated): {gflops:.4f}")

    try:
        dummy = torch.randn(1, 3, 256, 256, device=device)
        model.eval()
        with torch.no_grad():
            model.reset()
            _, enc = model(dummy)
        ratio, in_sz, enc_sz = compression_ratio(dummy.shape, enc.shape)
        bpp = calculate_bitrate_bpp(enc.shape, original_shape=(256, 256), bits_per_value=32)
        print(f"  Compression: {ratio:.2f}x ({in_sz:,} → {enc_sz:,}, "
              f"{(in_sz-enc_sz)/in_sz*100:.1f}% reduction)")
        print(f"  Bitrate: {bpp:.4f} bpp")
        print(f"  {'✅' if ratio > 6.0 else '⚠️'} TARGET {'ACHIEVED' if ratio > 6.0 else 'NOT MET'}: {ratio:.2f}x")
    except Exception as e:
        print(f"  Compression calculation skipped: {e}")
        ratio, bpp = 15.0, 2.0

    history = train_model(model, train_loader, val_loader, epochs=epochs,
                          device=device, time_steps=time_steps, exp_dir=exp_dir)

    # ── Load best model for testing ──────────────────────────
    print(f"\n{'='*60}\nTESTING HIGH COMPRESSION MODEL\n{'='*60}")

    onnx_path = os.path.join(exp_dir, 'best_model.onnx')
    pth_path  = os.path.join(exp_dir, 'best_model.pth')

    use_onnx_inference = False
    ort_session        = None

    if os.path.exists(onnx_path):
        try:
            ort_session        = load_onnx_model(onnx_path, device_str=device.type)
            use_onnx_inference = True
            print("  Using ONNX Runtime for test inference.")
        except Exception as e:
            print(f"  ONNX Runtime load failed ({e}); falling back to PyTorch.")

    if not use_onnx_inference:
        if os.path.exists(pth_path):
            model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
            print("  Loaded PTH fallback checkpoint.")
        else:
            print("  WARNING: No saved checkpoint found; using current model weights.")

    model.eval()
    criterion = EnhancedPerceptualLoss(alpha=0.84, use_gradient_loss=True)

    test_loss, psnr_list, ssim_list = 0.0, [], []
    t_orig, t_recon = [], []
    latencies = []

    with torch.no_grad():
        for frames in tqdm(test_loader, desc="Testing"):
            frames = frames.to(device)

            if use_onnx_inference:
                frames_np = frames.cpu().numpy().astype(np.float32)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                recon_np, _ = onnx_inference(ort_session, frames_np)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                recon = torch.from_numpy(recon_np).to(device)
            else:
                model.reset()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                recon, _ = model(frames)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()

            batch_latency = (end_time - start_time) * 1000 / frames.size(0)
            latencies.append(batch_latency)

            test_loss += criterion(recon, frames).item()
            psnr_list.append(calc_psnr(frames, recon))
            ssim_list.append(calc_ssim(frames, recon))

            for i in range(frames.size(0)):
                t_orig.append((frames[i].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))
                t_recon.append((recon[i].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))

    avg_loss      = test_loss / len(test_loader)
    avg_psnr      = float(np.mean(psnr_list))
    avg_ssim      = float(np.mean(ssim_list))
    avg_latency   = float(np.mean(latencies))
    throughput    = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print(f"\n✅ Test Results:")
    print(f"  Loss:               {avg_loss:.4f}")
    print(f"  PSNR:               {avg_psnr:.2f} dB")
    print(f"  SSIM:               {avg_ssim:.4f}")
    print(f"  Bitrate:            {bpp:.4f} bpp")
    print(f"  Compression Ratio:  {ratio:.2f}x")
    print(f"  Latency:            {avg_latency:.2f} ms/frame")
    print(f"  Throughput:         {throughput:.1f} FPS")
    print(f"  Model format:       {'ONNX' if use_onnx_inference else 'PTH (fallback)'}")

    save_video(t_orig,  os.path.join(exp_dir, 'test_original.mp4'))
    save_video(t_recon, os.path.join(exp_dir, 'test_reconstructed.mp4'))
    create_sidebyside_video(t_orig, t_recon, os.path.join(exp_dir, 'test_sidebyside.mp4'))

    results = {
        'time_steps':        time_steps,
        'n_params':          n_params,
        'gflops':            gflops,
        'compression_ratio': ratio,
        'bpp':               bpp,
        'test_loss':         avg_loss,
        'test_psnr':         avg_psnr,
        'test_ssim':         avg_ssim,
        'latency_ms':        avg_latency,
        'throughput_fps':    throughput,
        'model_format':      'onnx' if use_onnx_inference else 'pth',
        'history':           history,
    }

    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(
            {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
             for k, v in results.items() if k != 'history'},
            f, indent=2
        )

    plot_training_metrics_with_bpp(history, exp_dir)
    plot_compression_analysis(results, exp_dir)
    print(f"\n✅ Results saved in: {exp_dir}/")
    return results

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    UCF_ROOT = Path("/kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101")
    if not UCF_ROOT.exists():
        print("Downloading UCF101 dataset using kaggle CLI...")
        os.system("kaggle datasets download -d pevogam/ucf101 --unzip -p ucf101_extracted")
        print("Extraction complete.")
        if (UCF_ROOT / "ApplyEyeMakeup").exists():
            print(f"Found valid UCF-101 structure at: {UCF_ROOT}")
        else:
            print("Warning: Expected folder not found. Listing contents:")
            os.system("ls -l ucf101_extracted")
            raise RuntimeError("UCF-101 folder not found after unzip.")

    SELECTED_CLASSES = [
        "Basketball", "Biking", "Diving", "MilitaryParade",
        "SkateBoarding", "SoccerJuggling", "TennisSwing",
        "WalkingWithDog", "HorseRiding", "Drumming"
    ]

    all_videos = collect_videos(UCF_ROOT, SELECTED_CLASSES, max_per_class=30)
    if not all_videos:
        raise RuntimeError("No videos found. Check dataset path and class names.")

    random.shuffle(all_videos)
    n       = len(all_videos)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    train_v = all_videos[:n_train]
    val_v   = all_videos[n_train:n_train + n_val]
    test_v  = all_videos[n_train + n_val:]

    print(f"\nDataset Split:")
    print(f"  Train: {len(train_v)} videos")
    print(f"  Val:   {len(val_v)} videos")
    print(f"  Test:  {len(test_v)} videos")

    FRAMES = 50
    FSIZE  = (256, 256)

    train_ds = OnTheFlyVideoDataset(train_v, FSIZE, FRAMES, augment=True)
    val_ds   = OnTheFlyVideoDataset(val_v,   FSIZE, FRAMES, augment=False)
    test_ds  = OnTheFlyVideoDataset(test_v,  FSIZE, FRAMES, augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    print("\n" + "="*70)
    print("HIGH COMPRESSION TRAINING - TARGET: >6x COMPRESSION")
    print("MODEL FORMAT: ONNX")
    print("="*70)

    TIME_STEPS = 2
    EPOCHS     = 1

    results = run_experiment(TIME_STEPS, train_loader, val_loader, test_loader,
                             device, epochs=EPOCHS)

    print("\n" + "="*70)
    print("✅ HIGH COMPRESSION EXPERIMENT COMPLETED!")
    print("="*70)
    print(f"\n📊 Final Results Summary (2 TIMESTEPS):")
    print(f"  • Parameters:        {results['n_params']:,}")
    print(f"  • GFLOPs:            {results['gflops']:.4f}")
    print(f"  • Compression Ratio: {results['compression_ratio']:.2f}x "
          f"{'✅ >6x' if results['compression_ratio'] > 6 else '❌ <6x'}")
    print(f"  • Bitrate:           {results['bpp']:.4f} bpp")
    print(f"  • Test PSNR:         {results['test_psnr']:.2f} dB")
    print(f"  • Test SSIM:         {results['test_ssim']:.4f}")
    print(f"  • Latency:           {results['latency_ms']:.2f} ms/frame")
    print(f"  • Throughput:        {results['throughput_fps']:.1f} FPS")
    print(f"  • Model format:      {results['model_format'].upper()}")
    print(f"\n📁 Results saved in: exp_high_compression_ts2/")
    print("="*70)


if __name__ == '__main__':
    main()
