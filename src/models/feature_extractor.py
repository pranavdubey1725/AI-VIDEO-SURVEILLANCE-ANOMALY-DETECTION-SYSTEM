import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torchvision import models
from config import FEATURE_DIM, DEVICE


class ResNet50Extractor(nn.Module):
    """
    Pre-trained ResNet50 with the final classification layer removed.

    Input  : Tensor [B, 3, 224, 224]  — batch of frames
    Output : Tensor [B, 2048]          — feature vector per frame

    Why remove the last layer?
        ResNet50's final FC layer maps 2048 features → 1000 ImageNet classes.
        We don't want class predictions — we want the rich 2048-dim
        representation that lives just before that layer. That representation
        encodes what's visually happening in the frame (edges, shapes,
        textures, objects) without committing to a specific class label.

    Why freeze the weights?
        We're using ResNet50 purely as a feature extractor, not fine-tuning it.
        Freezing means no gradients flow through it during training — faster,
        uses less memory, and prevents destroying the pre-trained knowledge.
    """

    def __init__(self):
        super().__init__()

        # Load ResNet50 with ImageNet weights
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the final FC layer (classifier)
        # backbone.children() gives: [conv, bn, relu, maxpool, layer1, layer2,
        #                              layer3, layer4, avgpool, fc]
        # We keep everything except the last layer (fc)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze all parameters — we never update these weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor [B, 3, 224, 224]
        Returns:
            Tensor [B, 2048]
        """
        # Forward through frozen backbone → [B, 2048, 1, 1]
        out = self.features(x)
        # Remove spatial dimensions → [B, 2048]
        out = out.squeeze(-1).squeeze(-1)
        return out


def get_extractor(device: str = DEVICE) -> ResNet50Extractor:
    model = ResNet50Extractor()
    model = model.to(device)
    model.eval()   # always eval mode — no dropout, no batchnorm update
    return model


if __name__ == "__main__":
    device    = DEVICE
    extractor = get_extractor(device)

    # Quick shape check
    dummy = torch.randn(8, 3, 224, 224).to(device)
    with torch.no_grad():
        out = extractor(dummy)

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Expected     : [8, {FEATURE_DIM}]")
    print(f"Trainable params : {sum(p.numel() for p in extractor.parameters() if p.requires_grad)}")
    print(f"Total params     : {sum(p.numel() for p in extractor.parameters()):,}")
    print("ResNet50Extractor OK")
