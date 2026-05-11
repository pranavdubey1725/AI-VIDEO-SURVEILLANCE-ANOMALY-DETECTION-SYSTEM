"""
Training loop for AnomalyLSTM on pre-computed ResNet50 features.

What happens each epoch:
    1. Train on all clips with RankingLoss (anomaly score > normal score)
    2. Evaluate on val split — compute loss + AUC-ROC
    3. Save checkpoint if val loss improved (early stopping criterion)
    4. Adjust LR via ReduceLROnPlateau if val loss stagnates

Why AUC-ROC and not accuracy?
    Our val set has 76% normal clips. A model that predicts "normal" for
    everything gets 76% accuracy but is completely useless. AUC-ROC measures
    how well the model separates anomalous from normal clips regardless of
    the threshold — it's the right metric for imbalanced anomaly detection.

Expected training time: ~10-15 minutes per epoch on RTX 4060 (feature-based).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json
import time

from config import (
    CHECKPOINTS_DIR, LEARNING_RATE, NUM_EPOCHS, DEVICE, BATCH_SIZE
)
from src.models.lstm_model import get_model, RankingLoss
from src.dataset.feature_dataset import get_feature_dataloader


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for features, labels in tqdm(loader, desc="  Train", leave=False):
        features = features.to(device)   # [B, 16, 2048]
        labels   = labels.to(device)     # [B]

        optimizer.zero_grad()
        scores = model(features)          # [B, 1]
        loss   = loss_fn(scores, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss  = 0.0
    n_batches   = 0
    all_scores  = []
    all_labels  = []

    with torch.no_grad():
        for features, labels in tqdm(loader, desc="  Val  ", leave=False):
            features = features.to(device)
            labels   = labels.to(device)

            scores = model(features)      # [B, 1]
            loss   = loss_fn(scores, labels)

            total_loss += loss.item()
            n_batches  += 1

            all_scores.extend(scores.squeeze(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)

    # AUC-ROC: needs at least one positive and one negative sample
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)
    else:
        auc = 0.0

    return avg_loss, auc


def train(resume_from: Path = None):
    print("=" * 60)
    print("TRAINING AnomalyLSTM")
    print("=" * 60)
    print(f"Device     : {DEVICE}")
    print(f"Epochs     : {NUM_EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"LR         : {LEARNING_RATE}")
    print()

    # ── Data ──────────────────────────────────────────────────────
    print("Loading datasets...")
    train_loader = get_feature_dataloader("train", batch_size=BATCH_SIZE)
    val_loader   = get_feature_dataloader("val",   batch_size=BATCH_SIZE)
    print(f"Train batches : {len(train_loader):,}")
    print(f"Val   batches : {len(val_loader):,}")
    print()

    # ── Model / Optimizer / Scheduler ─────────────────────────────
    model     = get_model(DEVICE)
    loss_fn   = RankingLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ── Checkpoint setup ──────────────────────────────────────────
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_auc    = 0.0
    history     = []
    start_epoch = 1

    # ── Resume from checkpoint if provided ────────────────────────
    if resume_from and resume_from.exists():
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        best_auc    = ckpt["val_auc"]
        print(f"Resuming from epoch {start_epoch} | Best AUC so far: {best_auc:.4f}\n")

        # Load existing history if present
        history_path = CHECKPOINTS_DIR / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)

    print("Starting training...\n")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss           = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_auc    = evaluate(model, val_loader, loss_fn, DEVICE)

        scheduler.step(val_auc)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.0f}s"
        )

        # Save history
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_auc":    round(val_auc, 6),
            "lr":         current_lr,
        })

        # Save best checkpoint — track by AUC (val_loss saturates at 0)
        if val_auc > best_auc:
            best_auc  = val_auc
            ckpt_path = CHECKPOINTS_DIR / "best_model.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    val_loss,
                "val_auc":     val_auc,
            }, ckpt_path)
            print(f"  *** Saved best checkpoint (AUC={val_auc:.4f})")

        # Save latest checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":   val_loss,
                "val_auc":    val_auc,
            }, CHECKPOINTS_DIR / f"epoch_{epoch:02d}.pt")

    # Save training history to JSON
    history_path = CHECKPOINTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete.")
    print(f"Best AUC-ROC  : {best_auc:.4f}")
    print(f"Checkpoint    : {CHECKPOINTS_DIR / 'best_model.pt'}")
    print(f"History       : {history_path}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    resume_path = Path(args.resume) if args.resume else None
    train(resume_from=resume_path)
