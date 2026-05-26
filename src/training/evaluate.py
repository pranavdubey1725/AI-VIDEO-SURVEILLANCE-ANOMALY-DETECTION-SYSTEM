"""
Evaluates a trained AnomalyLSTM checkpoint on the test split.

Produces:
    - AUC-ROC score (primary metric)
    - Confusion matrix at the configured threshold
    - Per-category breakdown (which anomaly types are easiest/hardest)
    - Score distribution plots saved to outputs/
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
import json
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    CHECKPOINTS_DIR, OUTPUTS_DIR, DEVICE, BATCH_SIZE,
    ANOMALY_THRESHOLD, SPLITS_DIR
)
from src.models.lstm_model import get_model
from src.dataset.feature_dataset import get_feature_dataloader, FeatureDataset


def load_checkpoint(checkpoint_path: Path, device: str):
    model = get_model(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"  Saved val loss : {ckpt['val_loss']:.4f}")
    print(f"  Saved val AUC  : {ckpt['val_auc']:.4f}")
    return model


def get_scores_and_labels(model, loader, device):
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Scoring"):
            features = features.to(device)
            scores   = model(features)
            all_scores.extend(scores.squeeze(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_scores), np.array(all_labels)


def plot_roc_curve(labels, scores, output_dir: Path):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — AnomalyLSTM on Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = output_dir / "roc_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"ROC curve saved -> {path}")
    return auc


def plot_score_distribution(labels, scores, output_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(scores[labels == 0], bins=50, alpha=0.6, label="Normal",    color="steelblue")
    plt.hist(scores[labels == 1], bins=50, alpha=0.6, label="Anomalous", color="tomato")
    plt.axvline(ANOMALY_THRESHOLD, color="black", linestyle="--", label=f"Threshold={ANOMALY_THRESHOLD}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.title("Score Distribution: Normal vs Anomalous Clips")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "score_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Score distribution saved -> {path}")


def per_category_auc(scores, labels, split="test"):
    """
    Computes AUC per anomaly category by pairing each category's clips
    against all Normal clips. Each category only has one label type,
    so we can't compute AUC within a category — we compare it vs Normal.
    """
    csv_path = SPLITS_DIR / f"{split}.csv"
    df = pd.read_csv(csv_path)

    dataset = FeatureDataset(split)
    valid_indices = dataset._valid_indices

    df_valid = df.iloc[valid_indices].copy()
    df_valid["score"] = scores
    df_valid["label"] = labels

    # Pull out all Normal clips as the negative class
    normal_mask   = df_valid["label"] == 0
    normal_scores = df_valid.loc[normal_mask, "score"].values
    normal_labels = np.zeros(len(normal_scores), dtype=int)

    print("\nPer-category AUC-ROC (each anomaly type vs Normal):")
    print(f"  {'Category':<20} {'Clips':>8} {'AUC':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8}")

    results = {}
    anomaly_cats = [c for c in df_valid["category"].unique() if c != "Normal"]
    for cat in sorted(anomaly_cats):
        group       = df_valid[df_valid["category"] == cat]
        cat_scores  = group["score"].values
        cat_labels  = np.ones(len(cat_scores), dtype=int)

        combined_scores = np.concatenate([normal_scores, cat_scores])
        combined_labels = np.concatenate([normal_labels, cat_labels])

        cat_auc = roc_auc_score(combined_labels, combined_scores)
        results[cat] = cat_auc
        print(f"  {cat:<20} {len(group):>8,} {cat_auc:>8.4f}")

    return results


def evaluate(checkpoint_path: Path = None):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = CHECKPOINTS_DIR / "best_model.pt"

    print("=" * 60)
    print("EVALUATING AnomalyLSTM")
    print("=" * 60)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {DEVICE}")
    print()

    model = load_checkpoint(checkpoint_path, DEVICE)

    print("\nLoading test split...")
    test_loader = get_feature_dataloader("test", batch_size=BATCH_SIZE)
    print(f"Test batches: {len(test_loader):,}\n")

    scores, labels = get_scores_and_labels(model, test_loader, DEVICE)

    # ── Core metrics ──────────────────────────────────────────────
    auc = roc_auc_score(labels, scores)
    preds = (scores >= ANOMALY_THRESHOLD).astype(int)
    cm    = confusion_matrix(labels, preds)

    print("\n" + "=" * 60)
    print(f"Test AUC-ROC  : {auc:.4f}")
    print(f"Threshold     : {ANOMALY_THRESHOLD}")
    print()
    print("Confusion Matrix (rows=true, cols=pred):")
    print(f"               Pred Normal  Pred Anomaly")
    print(f"  True Normal  {cm[0,0]:>10,}  {cm[0,1]:>12,}")
    print(f"  True Anomaly {cm[1,0]:>10,}  {cm[1,1]:>12,}")
    print()
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=["Normal", "Anomalous"]))

    # ── Plots ─────────────────────────────────────────────────────
    plot_roc_curve(labels, scores, OUTPUTS_DIR)
    plot_score_distribution(labels, scores, OUTPUTS_DIR)

    # ── Per-category breakdown ────────────────────────────────────
    cat_aucs = per_category_auc(scores, labels)

    # ── Save results JSON ─────────────────────────────────────────
    results = {
        "checkpoint": str(checkpoint_path),
        "test_auc":   round(float(auc), 6),
        "threshold":  ANOMALY_THRESHOLD,
        "confusion_matrix": cm.tolist(),
        "per_category_auc": {k: round(float(v), 6) for k, v in cat_aucs.items()},
    }
    stem     = checkpoint_path.stem   # e.g. "best_model" or "best_model_aug"
    out_path = OUTPUTS_DIR / f"evaluation_results_{stem}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")
    print("=" * 60)

    return auc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file (default: checkpoints/best_model.pt)")
    args = parser.parse_args()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    evaluate(checkpoint_path=ckpt_path)
