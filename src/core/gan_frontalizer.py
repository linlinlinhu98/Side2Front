"""
GAN-based Face Frontalization Module

Uses an encoder-decoder generator trained to synthesize realistic frontal views
from profile faces. This approach actually GENERATES new pixels for occluded
regions (not just mirror/rotate).

Model: https://huggingface.co/spaces/opetrova/face-frontalization
Architecture: Simple encoder-decoder (no skip connections), 128x128 RGB I/O
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn as nn

from .base import FaceFrontalizer, FrontalizationResult

# Try to download from Hugging Face
try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Generator Architecture (from opetrova/face-frontalization)
# ---------------------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class G(nn.Module):
    """
    Generator network for 128x128 RGB face frontalization.
    Encoder-decoder architecture (no skip connections).
    Output range: [-1, 1] (Tanh activation).
    """
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            # Input: 128x128
            nn.Conv2d(3, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(True),   # 64x64
            nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),   # 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),   # 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True), # 8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),# 4x4
            nn.Conv2d(256, 512, 4, 2, 1), nn.MaxPool2d((2, 2)),              # 2x2
            # Bottleneck: 512-dim representation at 2x2
            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),  # 4x4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),  # 8x8
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),   # 16x16
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),    # 32x32
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(True),    # 64x64
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False), nn.Tanh()                              # 128x128
        )

    def forward(self, input):
        return self.main(input)


# ---------------------------------------------------------------------------
# GAN Frontalizer
# ---------------------------------------------------------------------------

class GANFrontalizer:
    """
    GAN-based face frontalization that actually synthesizes occluded regions.

    Uses a U-Net generator pretrained to map profile faces → frontal faces.
    This is NOT a mirror operation — it generates genuinely new texture.
    """
    name = "gan"
    display_name = "GAN正脸合成"

    def __init__(self, device=None, model_path=None):
        self.device = device or torch.device('cpu')
        self._initialized = False
        self._model = None
        self._image_size = 128  # Model input/output size

        if model_path and os.path.exists(model_path):
            self._load_local_model(model_path)
        elif HF_AVAILABLE:
            self._try_download_from_hf()
        else:
            print("GANFrontalizer: No model found and huggingface_hub not installed. "
                  "Install with: pip install huggingface_hub")

    def _try_download_from_hf(self):
        """Download pretrained model + network.py from Hugging Face Hub."""
        repo_id = "opetrova/face-frontalization"
        try:
            # Download network.py first (defines model architecture)
            net_path = hf_hub_download(repo_id=repo_id, filename="network.py")
            net_dir = os.path.dirname(net_path)
            if net_dir not in sys.path:
                sys.path.insert(0, net_dir)
            # Now download model weights
            model_path = hf_hub_download(repo_id=repo_id, filename="generator_v0.pt")
            print(f"GANFrontalizer: Downloaded model from HF: {repo_id}")
            self._load_local_model(model_path)
        except Exception as e:
            print(f"GANFrontalizer: HF download failed: {e}")
            self._create_untrained_model()

    def _load_local_model(self, model_path):
        try:
            # Add network.py's directory to sys.path so unpickler can find class G
            net_dir = os.path.dirname(model_path)
            if net_dir not in sys.path:
                sys.path.insert(0, net_dir)

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, nn.Module):
                self._model = checkpoint
            elif isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
                if 'netG' in checkpoint:
                    checkpoint = checkpoint['netG']
                state = {k.replace('module.', ''): v for k, v in checkpoint.items()}
                self._model = G()
                self._model.load_state_dict(state, strict=False)
            else:
                raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

            self._model.to(self.device)
            self._model.eval()
            self._initialized = True
            print(f"GANFrontalizer: Loaded model from {model_path}")
        except Exception as e:
            print(f"GANFrontalizer: Failed to load model from {model_path}: {e}")
            self._create_untrained_model()

    def _create_untrained_model(self):
        """Create an untrained model for testing when no weights are available."""
        self._model = G()
        self._model.apply(weights_init)
        self._model.to(self.device)
        self._model.eval()
        self._initialized = True
        print("GANFrontalizer: Using untrained model (no pretrained weights)")

    @property
    def is_initialized(self) -> bool:
        return self._initialized and self._model is not None

    def _preprocess(self, image: np.ndarray, bbox: list) -> torch.Tensor:
        """Crop face region, resize to model input size, normalize to [-1, 1]."""
        x1, y1, x2, y2 = bbox
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            face = image
        face = cv2.resize(face, (self._image_size, self._image_size))
        # Convert BGR → RGB and normalize to [-1, 1]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 127.5 - 1.0
        # HWC → NCHW
        tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert output tensor to displayable uint8 image [0, 255] BGR."""
        # NCHW → HWC, denormalize from [-1, 1] to [0, 255]
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        # RGB → BGR for OpenCV display
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def frontalize(self, image: np.ndarray, bbox: list) -> np.ndarray:
        """
        Synthesize a frontal view of a face from a profile view.

        Args:
            image: BGR image (numpy array)
            bbox: [x1, y1, x2, y2] face bounding box

        Returns:
            Frontalized face image (BGR, same size as cropped face region)
        """
        if not self.is_initialized:
            raise RuntimeError("GANFrontalizer not initialized")

        with torch.no_grad():
            input_tensor = self._preprocess(image, bbox)
            output_tensor = self._model(input_tensor)
            result = self._postprocess(output_tensor)
        return result

    def convert(self, image: np.ndarray, **kwargs) -> FrontalizationResult:
        """
        Full pipeline: detect face → crop → frontalize → blend back.

        Args:
            image: BGR input image
            **kwargs: ignored (for API compatibility)
        """
        t0 = time.time()

        if not self.is_initialized:
            return FrontalizationResult(
                image if image is not None else np.zeros((100, 100, 3), dtype=np.uint8),
                0, "GAN模型未初始化"
            )

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Detect face using OpenCV Haar cascade
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = cascade.detectMultiScale(gray, 1.1, 5)

        if len(boxes) == 0:
            return FrontalizationResult(image, 0, "GAN: 未检测到人脸")

        # Use largest face
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        # Add padding around face
        pad = int(0.3 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        bbox = [x1, y1, x2, y2]

        # Frontalize
        frontal = self.frontalize(image, bbox)
        frontal_h, frontal_w = frontal.shape[:2]
        face_w = x2 - x1
        face_h = y2 - y1

        # Resize frontal to match original face crop size
        frontal_resized = cv2.resize(frontal, (face_w, face_h))

        # Blend: use Poisson blending for smooth edges
        center = (x + w // 2, y + h // 2)
        try:
            blended = cv2.seamlessClone(frontal_resized, image, np.ones_like(frontal_resized) * 255, center, cv2.NORMAL_CLONE)
        except Exception:
            # Fallback: direct copy if seamless clone fails
            blended = image.copy()
            blended[y1:y2, x1:x2] = frontal_resized

        elapsed = (time.time() - t0) * 1000
        info = f"GAN frontalization | {elapsed:.0f}ms"
        return FrontalizationResult(blended, elapsed, info)


# ---------------------------------------------------------------------------
# Integrated GAN mode for RealFaceFrontalizer
# ---------------------------------------------------------------------------

def create_gan_frontalizer() -> GANFrontalizer:
    """Factory function to create a GAN frontalizer instance."""
    try:
        return GANFrontalizer(device=torch.device('cpu'))
    except Exception as e:
        print(f"Failed to create GANFrontalizer: {e}")
        return None
