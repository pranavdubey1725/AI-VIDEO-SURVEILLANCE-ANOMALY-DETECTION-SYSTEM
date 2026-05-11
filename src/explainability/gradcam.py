"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet50.

What it does:
    Produces a heatmap showing which spatial regions of a frame were
    most important when generating the 2048-dim feature vector that
    the LSTM used to score the clip.

Why it works here even though ResNet50 is frozen:
    Grad-CAM only needs a forward pass + one backward pass to compute
    gradients. It does NOT update any weights. We temporarily allow
    gradient flow through ResNet50 just for the heatmap computation,
    then discard it.

How it works:
    1. Hook into ResNet50's last conv layer (layer4)
    2. Forward pass: frame -> layer4 activations -> 2048-dim features
    3. Backward pass: gradient of the feature norm w.r.t. layer4 activations
    4. Weight each activation channel by its gradient (global average)
    5. ReLU + resize -> heatmap in [0,1] matching input frame size

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from config import DEVICE, FRAME_SIZE
from src.models.feature_extractor import get_extractor


TRANSFORM = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class GradCAM:
    """
    Grad-CAM for ResNet50Extractor, targeting the last conv layer (layer4).

    Usage:
        gradcam = GradCAM(extractor)
        heatmap = gradcam.compute(pil_image)          # [H, W] float in [0,1]
        overlay = gradcam.overlay(pil_image, heatmap) # PIL Image with heatmap
    """

    def __init__(self, extractor):
        self.extractor = extractor
        self.device    = next(extractor.parameters()).device

        self._activations = None
        self._gradients   = None

        # Hook into layer4 — the last conv block before avgpool
        # extractor.features is nn.Sequential of ResNet50 children
        # Index 7 = layer4 in the standard ResNet50 child order:
        # [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
        #    0      1    2     3        4       5       6       7       8
        self._target_layer = extractor.features[7]
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._target_layer.register_forward_hook(forward_hook)
        self._target_layer.register_full_backward_hook(backward_hook)

    def compute(self, frame) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single frame.

        Args:
            frame : PIL Image or numpy [H, W, 3]
        Returns:
            heatmap : numpy [H, W] float32 in [0, 1]
        """
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)

        orig_w, orig_h = frame.size

        # Temporarily enable gradients through the frozen backbone
        original_grad_states = {}
        for name, param in self.extractor.named_parameters():
            original_grad_states[name] = param.requires_grad
            param.requires_grad_(True)

        self.extractor.train()   # needed for gradient flow in some BN configs

        x = TRANSFORM(frame).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        x.requires_grad_(True)

        # Forward pass
        features = self.extractor(x)   # [1, 2048]

        # Scalar target: L2 norm of the feature vector
        # Maximising the feature norm → which regions contributed most
        target = features.norm()
        self.extractor.zero_grad()
        target.backward()

        # Restore frozen state
        for name, param in self.extractor.named_parameters():
            param.requires_grad_(original_grad_states[name])
        self.extractor.eval()

        # Grad-CAM formula:
        # weights[c] = global_avg_pool(gradients[c])   shape: [C]
        # cam = ReLU( sum_c( weights[c] * activations[c] ) )
        gradients   = self._gradients[0]    # [C, H', W']  e.g. [2048, 7, 7]
        activations = self._activations[0]  # [C, H', W']

        weights = gradients.mean(dim=(1, 2))              # [C]
        cam     = (weights[:, None, None] * activations).sum(0)  # [H', W']
        cam     = torch.relu(cam)

        # Normalise to [0, 1]
        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        # Resize to original frame size
        heatmap = cv2.resize(cam_np, (orig_w, orig_h))
        return heatmap.astype(np.float32)

    def overlay(self, frame, heatmap: np.ndarray,
                alpha: float = 0.4) -> Image.Image:
        """
        Overlay the heatmap on the original frame using a jet colormap.

        Args:
            frame   : PIL Image or numpy [H, W, 3] (RGB)
            heatmap : numpy [H, W] float32 in [0, 1]
            alpha   : heatmap opacity (0=invisible, 1=opaque)
        Returns:
            PIL Image with heatmap overlay
        """
        if isinstance(frame, Image.Image):
            frame_np = np.array(frame)
        else:
            frame_np = frame.copy()

        # Convert heatmap to BGR colormap (jet: blue=low, red=high)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = (frame_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
        return Image.fromarray(overlay)

    def compute_and_overlay(self, frame, alpha: float = 0.4) -> tuple:
        """Convenience: returns (heatmap, overlay_image) in one call."""
        heatmap = self.compute(frame)
        overlay = self.overlay(frame, heatmap, alpha=alpha)
        return heatmap, overlay


def get_gradcam(extractor=None) -> GradCAM:
    if extractor is None:
        extractor = get_extractor(DEVICE)
    return GradCAM(extractor)


if __name__ == "__main__":
    from pathlib import Path

    print("Testing Grad-CAM...")
    extractor = get_extractor(DEVICE)
    gradcam   = GradCAM(extractor)

    # Create a test frame with a simple pattern
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[150:330, 220:420] = [180, 100, 80]   # a coloured rectangle
    test_pil = Image.fromarray(test_frame)

    heatmap, overlay = gradcam.compute_and_overlay(test_pil)

    print(f"Heatmap shape : {heatmap.shape}")
    print(f"Heatmap range : [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print(f"Overlay size  : {overlay.size}")

    # Save test output
    from config import OUTPUTS_DIR
    OUTPUTS_DIR.mkdir(exist_ok=True)
    overlay.save(OUTPUTS_DIR / "gradcam_test.png")
    print(f"Test overlay saved -> outputs/gradcam_test.png")
    print("GradCAM OK")
