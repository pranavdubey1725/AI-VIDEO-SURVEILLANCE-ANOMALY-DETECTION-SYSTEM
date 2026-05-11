import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from config import FEATURE_DIM, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT, DEVICE


class AnomalyLSTM(nn.Module):
    """
    LSTM-based anomaly scorer for video clips.

    Input  : [batch, CLIP_LENGTH, FEATURE_DIM]  — sequence of frame features
    Output : [batch, 1]                          — anomaly score in [0, 1]

    Architecture:
        Stacked LSTM → last hidden state → FC layer → Sigmoid
    """

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size   = FEATURE_DIM,    # 2048 — ResNet50 feature dim
            hidden_size  = LSTM_HIDDEN,    # 256  — internal memory size
            num_layers   = LSTM_LAYERS,    # 2    — stacked LSTMs
            batch_first  = True,           # input shape: [batch, seq, features]
            dropout      = DROPOUT if LSTM_LAYERS > 1 else 0.0,
            bidirectional= False           # unidirectional: past → future
        )

        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(LSTM_HIDDEN, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()           # output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : FloatTensor [batch, seq_len, feature_dim]
        Returns:
            scores : FloatTensor [batch, 1]  — anomaly probability per clip
        """
        # lstm_out: [batch, seq_len, hidden_size] — output at every timestep
        # hidden:   ([num_layers, batch, hidden], [num_layers, batch, hidden])
        lstm_out, (hidden, _) = self.lstm(x)

        # Take the last hidden state from the top LSTM layer
        # hidden[-1] = [batch, hidden_size] — top layer's final state
        last_hidden = hidden[-1]

        scores = self.classifier(last_hidden)  # [batch, 1]
        return scores


class RankingLoss(nn.Module):
    """
    Weakly-supervised ranking loss for anomaly detection.

    Enforces: score(anomaly) > score(normal) by a margin of 1.

    loss = mean( max(0,  1 - score_anom + score_norm) )

    Why this works with weak labels:
        We don't know WHICH frames are anomalous in a video.
        But we DO know anomalous clips should score higher than normal ones.
        The ranking loss enforces this ordering without frame-level labels.

    This is the core insight from Sultani et al. (CVPR 2018) — the landmark
    paper that defined the UCF-Crime benchmark.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores : [batch, 1]  — model output, values in [0,1]
            labels : [batch]     — 0 for normal, 1 for anomalous
        Returns:
            loss : scalar tensor
        """
        scores  = scores.squeeze(1)               # [batch]
        anom_mask  = (labels == 1)
        normal_mask = (labels == 0)

        anom_scores   = scores[anom_mask]
        normal_scores = scores[normal_mask]

        # Need at least one of each class in the batch
        if len(anom_scores) == 0 or len(normal_scores) == 0:
            return torch.tensor(0.0, requires_grad=True, device=scores.device)

        # Broadcast: compare every anomaly score against every normal score
        # anom:   [n_anom, 1]   normal: [1, n_normal]
        # diff:   [n_anom, n_normal]
        diff = anom_scores.unsqueeze(1) - normal_scores.unsqueeze(0)
        loss = torch.clamp(self.margin - diff, min=0.0)
        return loss.mean()


def get_model(device: str = DEVICE) -> AnomalyLSTM:
    model = AnomalyLSTM().to(device)
    return model


if __name__ == "__main__":
    device = DEVICE
    model  = get_model(device)

    # Shape check
    dummy_input = torch.randn(8, 16, FEATURE_DIM).to(device)
    scores      = model(dummy_input)

    print(f"Input  shape : {dummy_input.shape}")
    print(f"Output shape : {scores.shape}")
    print(f"Score range  : [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"\nModel parameters:")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total      : {total:,}")
    print(f"  Trainable  : {trainable:,}")

    # Loss check
    labels = torch.randint(0, 2, (8,)).to(device)
    loss_fn = RankingLoss()
    loss    = loss_fn(scores, labels)
    print(f"\nRanking loss (random scores): {loss.item():.4f}")
    print("AnomalyLSTM OK")
