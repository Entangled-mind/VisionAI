"""VisionAI - Milestone 9: Quantitative Model Evaluation Module

Implements evaluation metrics for:
1. Face Recognition:
   - Genuine Match Distribution (intra-class similarity)
   - Impostor Match Distribution (inter-class similarity)
   - False Acceptance Rate (FAR) vs. False Rejection Rate (FRR)
   - Equal Error Rate (EER) approximation
   - Optimal Cosine Similarity Threshold Selection
2. Object Detection:
   - Bounding Box Intersection over Union (IoU)
   - Precision, Recall, and F1-Score calculations
   - Confusion Matrix breakdown
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import compute_iou
from src.face_recognition import compute_cosine_similarity


@dataclass
class FaceEvalReport:
    """Evaluation metrics for face recognition."""
    threshold: float
    genuine_matches: int
    impostor_matches: int
    false_accepts: int
    false_rejects: int
    far: float  # False Acceptance Rate
    frr: float  # False Rejection Rate
    accuracy: float


def evaluate_face_thresholds(
    genuine_pairs: List[Tuple[np.ndarray, np.ndarray]],
    impostor_pairs: List[Tuple[np.ndarray, np.ndarray]],
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Evaluates FAR, FRR, and Accuracy across a range of cosine similarity thresholds.

    Definitions:
        - Genuine Pair: Two embeddings of the SAME person. Should have similarity >= threshold.
          If similarity < threshold -> False Rejection (Type II error).
        - Impostor Pair: Two embeddings of DIFFERENT people. Should have similarity < threshold.
          If similarity >= threshold -> False Acceptance (Type I error).

    Args:
        genuine_pairs: List of (emb_A, emb_B) pairs for same identity.
        impostor_pairs: List of (emb_A, emb_B) pairs for different identities.
        thresholds: List of similarity thresholds to test (default: 0.10 to 0.90 in steps of 0.05).

    Returns:
        pd.DataFrame: Table of evaluation metrics per threshold.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]

    # Compute raw similarity scores
    genuine_scores = [compute_cosine_similarity(a, b) for a, b in genuine_pairs]
    impostor_scores = [compute_cosine_similarity(a, b) for a, b in impostor_pairs]

    n_gen = max(1, len(genuine_scores))
    n_imp = max(1, len(impostor_scores))

    rows = []
    for thresh in thresholds:
        # False Rejections: genuine pairs that fell BELOW threshold
        false_rejects = sum(1 for s in genuine_scores if s < thresh)
        frr = false_rejects / n_gen

        # False Acceptances: impostor pairs that scored ABOVE threshold
        false_accepts = sum(1 for s in impostor_scores if s >= thresh)
        far = false_accepts / n_imp

        # Overall Accuracy
        correct_gen = sum(1 for s in genuine_scores if s >= thresh)
        correct_imp = sum(1 for s in impostor_scores if s < thresh)
        accuracy = (correct_gen + correct_imp) / (n_gen + n_imp)

        rows.append({
            "Threshold": thresh,
            "FAR (%)": round(far * 100, 2),
            "FRR (%)": round(frr * 100, 2),
            "Accuracy (%)": round(accuracy * 100, 2),
            "FAR_raw": far,
            "FRR_raw": frr,
        })

    return pd.DataFrame(rows)


def evaluate_detection_boxes(
    pred_boxes: List[Tuple[int, int, int, int]],
    gt_boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculates Precision, Recall, and F1-Score for object bounding boxes against ground truth.

    Args:
        pred_boxes: List of predicted (x, y, w, h) boxes.
        gt_boxes: List of ground-truth (x, y, w, h) boxes.
        iou_threshold: Minimum IoU to consider a prediction a True Positive (default: 0.5).

    Returns:
        Dict: {"precision": float, "recall": float, "f1_score": float, "tp": int, "fp": int, "fn": int}
    """
    matched_gt = set()
    tp = 0
    fp = 0

    for p_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    """Demonstrates quantitative evaluation on synthetic benchmark pairs."""
    print("\n" + "=" * 60)
    print("      VisionAI - Milestone 9: Model Evaluation Benchmark")
    print("=" * 60)

    # 1. Benchmark Face Recognition Thresholds
    print("\n[Face Recognition Threshold Evaluation]")
    np.random.seed(42)

    # Create synthetic genuine pairs (high cosine similarity ~0.65 to 0.95)
    genuine_pairs = []
    for _ in range(50):
        base = np.random.randn(128)
        base /= np.linalg.norm(base)
        noise = np.random.randn(128) * 0.3
        perturbed = base + noise
        perturbed /= np.linalg.norm(perturbed)
        genuine_pairs.append((base, perturbed))

    # Create synthetic impostor pairs (low cosine similarity around -0.1 to 0.25)
    impostor_pairs = []
    for _ in range(50):
        v1 = np.random.randn(128)
        v1 /= np.linalg.norm(v1)
        v2 = np.random.randn(128)
        v2 /= np.linalg.norm(v2)
        impostor_pairs.append((v1, v2))

    eval_df = evaluate_face_thresholds(genuine_pairs, impostor_pairs)
    print("\nThreshold Tradeoff Table:")
    print(eval_df[["Threshold", "FAR (%)", "FRR (%)", "Accuracy (%)"]].to_string(index=False))

    # Find optimal threshold where |FAR - FRR| is minimized (Equal Error Rate point)
    eval_df["EER_diff"] = (eval_df["FAR_raw"] - eval_df["FRR_raw"]).abs()
    best_row = eval_df.sort_values("EER_diff").iloc[0]
    print(f"\nRecommended Optimal Threshold: {best_row['Threshold']} (Accuracy: {best_row['Accuracy (%)']}%, FAR: {best_row['FAR (%)']}%, FRR: {best_row['FRR (%)']}%)")

    # 2. Benchmark Object Detection IoU
    print("\n" + "-" * 60)
    print("[Object Detection IoU & Precision/Recall Benchmark]")
    gt_sample = [(50, 50, 100, 100), (200, 200, 80, 80)]
    pred_sample = [(52, 48, 98, 102), (195, 205, 82, 78), (400, 400, 50, 50)]  # 2 correct, 1 false positive

    box_metrics = evaluate_detection_boxes(pred_sample, gt_sample, iou_threshold=0.5)
    print(f"  * True Positives (TP) : {box_metrics['tp']}")
    print(f"  * False Positives (FP): {box_metrics['fp']}")
    print(f"  * False Negatives (FN): {box_metrics['fn']}")
    print(f"  * Precision           : {box_metrics['precision']*100:.1f}%")
    print(f"  * Recall              : {box_metrics['recall']*100:.1f}%")
    print(f"  * F1-Score            : {box_metrics['f1_score']*100:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
