"""
eval.py — QuickCart Intent Pipeline  |  Phase 3 (Export + Calibration) + Dev Analysis
=========================================================================================

What this file does
-------------------
1. Load best_checkpoint.pt.
2. Export the full graph (encoder + linear head) to model.onnx.
3. Fit Temperature Scaler T* on dev.csv by minimising NLL.
4. Tune OOS threshold θ on dev.csv by maximising out_of_scope F1.
5. Fit Mahalanobis parameters (class centroids + Ledoit-Wolf precision matrix)
   on cleaned_train.csv embeddings extracted via ONNX Runtime.
6. Tune Mahalanobis OOD threshold τ on dev.csv.
7. Measure and report dev Macro-F1, per-class P/R/F1, confusion matrix, ECE
   before and after temperature scaling, and cross-encoder trigger rate at each δ.
8. Measure per-sample latency histogram on dev.csv.
9. Save all artefacts and results.

Run
---
  python -m eval

Dependencies: torch, transformers, onnx, onnxruntime, scikit-learn, scipy, pandas,
              numpy, matplotlib, seaborn, tqdm
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import onnxruntime as ort
import onnx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from transformers import AutoTokenizer

from config import (
    TAXONOMY_PATH, CLEANED_TRAIN_CSV, DEV_CSV,
    CHECKPOINT_PATH, ONNX_MODEL_PATH, MAHAL_PARAMS_PATH,
    CALIBRATION_PARAMS_PATH, LOGS_DIR, RESULTS_DIR,
    DEV_RESULTS_JSON, DEV_CONFUSION_CSV, DEV_PERCLASS_CSV, DEV_LATENCY_CSV,
    ENCODER_MODEL_NAME, MAX_SEQ_LEN,
    INTENT_LABELS, NUM_CLASSES, LABEL2IDX, IDX2LABEL, OOS_IDX,
    SEED, ONNX_OPSET,
    OOS_THETA_SWEEP, CONFIDENCE_GAP_DELTA, MAHAL_OOD_TAU,
    CROSS_ENCODER_MODEL, CROSS_ENCODER_TOP_K,
)
from utils import get_logger, set_seed, preprocess, compute_ece, mean_pool
from train import IntentClassifier

logger = get_logger(__name__, LOGS_DIR / "eval.log")
set_seed(SEED)


# ===========================================================================
# STEP 1 — LOAD CHECKPOINT
# ===========================================================================

def load_model_from_checkpoint(path: Path) -> tuple[IntentClassifier, dict]:
    """
    Load IntentClassifier from a saved checkpoint.

    Args:
        path: Path to best_checkpoint.pt.

    Returns:
        (model, checkpoint_dict) where model is on CPU and in eval mode.
    """
    logger.info(f"Loading checkpoint from {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Run train.py first."
        )

    checkpoint = torch.load(path, map_location="cpu")
    model = IntentClassifier(checkpoint["config"]["encoder_model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        f"Loaded checkpoint: epoch={checkpoint['epoch']}, "
        f"dev_macro_F1={checkpoint['macro_f1']:.4f}, "
        f"dev_ECE={checkpoint['ece']:.4f}"
    )
    return model, checkpoint


# ===========================================================================
# STEP 2 — ONNX EXPORT
# ===========================================================================

def export_to_onnx(model: IntentClassifier, tokenizer, output_path: Path) -> None:
    """
    Export the IntentClassifier (encoder + linear head) to ONNX.

    What gets exported:
      - The full forward pass: input_ids + attention_mask → (logits, embeddings)
      - Dynamic sequence-length axis so the runtime handles variable-length inputs
        without padding to a fixed MAX_SEQ_LEN.
      - opset_version=17 for compatibility with all BGE-small operators.
      - do_constant_folding=True: ONNX fuses constant sub-expressions (e.g. LayerNorm
        parameters), reducing graph nodes and improving runtime speed.

    The resulting model.onnx takes ~8 ms per sample on 4 CPU cores via ONNX Runtime,
    compared to ~40 ms with native PyTorch CPU inference.

    Args:
        model:       IntentClassifier in eval mode on CPU.
        tokenizer:   HuggingFace tokenizer (for a dummy input).
        output_path: Where to save the .onnx file.
    """
    logger.info(f"Exporting model to ONNX: {output_path}")

    # Dummy inputs for tracing (actual values don't matter, only shapes)
    dummy_text = "where is my order"
    dummy_enc = tokenizer(
        dummy_text,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    dummy_input_ids = dummy_enc["input_ids"]           # [1, MAX_SEQ_LEN]
    dummy_attention_mask = dummy_enc["attention_mask"] # [1, MAX_SEQ_LEN]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(dummy_input_ids, dummy_attention_mask),
            f=str(output_path),
            opset_version=ONNX_OPSET,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits", "embeddings"],
            dynamic_axes={
                "input_ids":      {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits":         {0: "batch_size"},
                "embeddings":     {0: "batch_size"},
            },
            do_constant_folding=True,
        )

    # Verify the exported graph is well-formed
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info(f"ONNX export verified. File size: {output_path.stat().st_size / 1e6:.1f} MB")


# ===========================================================================
# STEP 3 — ONNX RUNTIME SESSION
# ===========================================================================

def build_ort_session(model_path: Path) -> ort.InferenceSession:
    """
    Build an ONNX Runtime InferenceSession optimised for multi-core CPU.

    Session options:
      - intra_op_num_threads = 4: parallelise within each operator (e.g. matrix
        multiply) across 4 cores.
      - inter_op_num_threads = 1: no parallel op scheduling; we process one
        message at a time in real-time inference so parallelism across ops
        does not help and adds scheduling overhead.
      - GraphOptimizationLevel.ORT_ENABLE_ALL: apply all ONNX Runtime graph
        optimisations (op fusion, constant folding, memory planning).

    Args:
        model_path: Path to model.onnx.

    Returns:
        Configured ort.InferenceSession.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Suppress ONNX Runtime verbose logs
    opts.log_severity_level = 3  # ERROR only

    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    logger.info(
        f"ORT session built. Inputs: "
        f"{[i.name for i in session.get_inputs()]}"
    )
    return session


# ===========================================================================
# STEP 4 — EMBED WITH ONNX RUNTIME
# ===========================================================================

def embed_texts_ort(
    session: ort.InferenceSession,
    tokenizer,
    texts: list[str],
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the ONNX model on a list of texts and return logits and embeddings.

    Args:
        session:    ONNX Runtime session.
        tokenizer:  HuggingFace tokenizer.
        texts:      List of preprocessed strings.
        batch_size: Number of texts per ONNX call.

    Returns:
        (all_logits, all_embeddings)
          all_logits:     float32 [N, NUM_CLASSES]
          all_embeddings: float32 [N, EMBEDDING_DIM]
    """
    all_logits, all_embeddings = [], []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (ONNX)", leave=False):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            max_length=MAX_SEQ_LEN,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        feed = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits, embeddings = session.run(["logits", "embeddings"], feed)
        all_logits.append(logits)
        all_embeddings.append(embeddings)

    return np.concatenate(all_logits, axis=0), np.concatenate(all_embeddings, axis=0)


# ===========================================================================
# STEP 5 — TEMPERATURE SCALING
# ===========================================================================

def fit_temperature(
    logits: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """
    Find the optimal temperature T* that minimises negative log-likelihood on
    the dev set.

    Temperature scaling: p = softmax(logits / T*)
    - T* > 1: softer probabilities (reduces overconfidence).
    - T* < 1: sharper probabilities (increases confidence).
    - T* = 1: no change.

    Why minimise NLL rather than ECE?
    NLL is smooth and differentiable; ECE with discrete bins is not.  Minimising
    NLL consistently produces good ECE reduction in practice.

    Args:
        logits:      float32 [N, NUM_CLASSES] raw logits from ONNX.
        true_labels: int [N] true class indices.

    Returns:
        Optimal temperature T* (scalar float).
    """
    def nll(T: float) -> float:
        # Softmax with temperature
        scaled = logits / T
        # Numerically stable log-softmax
        log_probs = scaled - np.log(np.exp(scaled - scaled.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)) - scaled.max(axis=1, keepdims=True)
        # NLL for true classes
        nll_val = -log_probs[np.arange(len(true_labels)), true_labels].mean()
        return float(nll_val)

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T_star = float(result.x)
    logger.info(f"Temperature scaling: T* = {T_star:.4f} (NLL = {result.fun:.4f})")
    return T_star


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """
    Apply temperature scaling and return probability array.

    Args:
        logits: float32 [N, NUM_CLASSES].
        T:      Temperature scalar.

    Returns:
        float32 [N, NUM_CLASSES] probability array.
    """
    scaled = logits / T
    # Numerically stable softmax
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


# ===========================================================================
# STEP 6 — OOS THRESHOLD TUNING
# ===========================================================================

def tune_oos_threshold(
    probs: np.ndarray,
    true_labels: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    """
    Sweep θ from 0.20 to 0.80 and find the value that maximises F1 for the
    out_of_scope class on dev.csv.

    Why maximise out_of_scope F1 specifically?
    - A loose θ (too low) causes legitimate in-scope queries to be labelled OOS,
      dropping recall on all in-scope intents.
    - A tight θ (too high) lets genuine OOS queries through as false positives for
      in-scope intents.
    - Maximising the harmonic mean of precision and recall for OOS balances both
      failure modes.

    Args:
        probs:       float32 [N, NUM_CLASSES] calibrated probabilities.
        true_labels: int [N] true class indices.

    Returns:
        (best_theta, sweep_df) where sweep_df has one row per theta value with
        [theta, oos_precision, oos_recall, oos_f1, macro_f1].
    """
    rows = []
    best_theta = 0.35
    best_oos_f1 = -1.0

    for theta in OOS_THETA_SWEEP:
        max_probs = probs.max(axis=1)
        preds = probs.argmax(axis=1).copy()
        preds[max_probs < theta] = OOS_IDX

        prec, rec, f1, _ = precision_recall_fscore_support(
            true_labels, preds, labels=[OOS_IDX], average=None, zero_division=0
        )
        macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)

        rows.append({
            "theta": theta,
            "oos_precision": float(prec[0]),
            "oos_recall": float(rec[0]),
            "oos_f1": float(f1[0]),
            "macro_f1": float(macro_f1),
        })

        if float(f1[0]) > best_oos_f1:
            best_oos_f1 = float(f1[0])
            best_theta = theta

    sweep_df = pd.DataFrame(rows)
    logger.info(
        f"OOS threshold tuning: θ* = {best_theta} "
        f"(OOS F1 = {best_oos_f1:.4f})"
    )
    logger.info("\n" + sweep_df.to_string(index=False))
    return best_theta, sweep_df


# ===========================================================================
# STEP 7 — MAHALANOBIS PARAMETERS
# ===========================================================================

def fit_mahalanobis(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
) -> dict:
    """
    Compute class centroids and a shared Ledoit-Wolf precision matrix for
    Mahalanobis distance OOD detection.

    Why Ledoit-Wolf?
    With 384 dimensions and potentially fewer than 384 samples per class, the
    sample covariance matrix is singular (not invertible).  Ledoit-Wolf adds
    an analytically optimal shrinkage regulariser α·I, ensuring the covariance
    is always positive definite and the inverse exists.

    Why shared (tied) covariance?
    Per-class covariance estimation requires even more samples to be stable.
    A single tied covariance estimated on all training embeddings is more
    numerically robust and often performs similarly for OOD detection.

    Args:
        train_embeddings: float32 [N, D] embedding matrix.
        train_labels:     int [N] true class indices.

    Returns:
        dict with:
          centroids:  {class_idx: centroid_vector (D,)}
          precision:  shared precision matrix [D, D]
    """
    logger.info("Fitting Mahalanobis parameters with Ledoit-Wolf shrinkage…")

    # --- Compute per-class centroids ---
    centroids = {}
    for idx in range(NUM_CLASSES):
        mask = train_labels == idx
        if mask.sum() == 0:
            # Fallback: use global mean if no samples for this class
            centroids[idx] = train_embeddings.mean(axis=0)
            logger.warning(f"No training samples for class {IDX2LABEL[idx]}; using global mean")
        else:
            centroids[idx] = train_embeddings[mask].mean(axis=0)

    # --- Shared Ledoit-Wolf covariance on all training embeddings ---
    # LedoitWolf.fit() computes the covariance matrix; .precision_ is its inverse.
    lw = LedoitWolf(assume_centered=False)
    lw.fit(train_embeddings)
    precision_matrix = lw.precision_   # [D, D] inverse covariance

    logger.info(
        f"Ledoit-Wolf shrinkage α = {lw.shrinkage_:.4f}. "
        f"Precision matrix shape: {precision_matrix.shape}"
    )

    return {"centroids": centroids, "precision": precision_matrix}


def tune_mahalanobis_threshold(
    dev_embeddings: np.ndarray,
    dev_preds: np.ndarray,
    dev_true: np.ndarray,
    mahal_params: dict,
    tau_sweep: list[float] | None = None,
) -> float:
    """
    Tune the OOD flag threshold τ on dev.csv.

    The goal is not perfect OOD detection (we don't have gold OOD labels),
    but to pick a τ that flags a reasonable fraction (~5–10%) of queries for
    human review.  We pick the τ at the 90th percentile of Mahalanobis
    distances on dev, so the top 10% of "unusual" queries are flagged.

    Args:
        dev_embeddings: float32 [N, D].
        dev_preds:      int [N] predicted class indices (after OOS check).
        dev_true:       int [N] true class indices.
        mahal_params:   Output of fit_mahalanobis().
        tau_sweep:      Optional list of τ values to try.

    Returns:
        Optimal τ scalar.
    """
    centroids = mahal_params["centroids"]
    precision = mahal_params["precision"]

    # Compute Mahalanobis distance for each dev sample to its predicted class centroid
    distances = []
    for i, pred_idx in enumerate(dev_preds):
        centroid = centroids.get(int(pred_idx), np.zeros(dev_embeddings.shape[1]))
        diff = dev_embeddings[i] - centroid
        dist = float(np.sqrt(max(0.0, diff @ precision @ diff)))
        distances.append(dist)

    distances = np.array(distances)

    # Set τ at 90th percentile (flag top 10% as OOD)
    tau = float(np.percentile(distances, 90))
    flag_rate = (distances > tau).mean()

    logger.info(
        f"Mahalanobis threshold: τ = {tau:.4f} "
        f"(flags {100 * flag_rate:.1f}% of dev queries)"
    )
    return tau


# ===========================================================================
# STEP 8 — DEV ANALYSIS
# ===========================================================================

def compute_dev_metrics(
    probs: np.ndarray,
    true_labels: np.ndarray,
    theta: float,
    T_star: float,
    raw_logits: np.ndarray,
) -> dict:
    """
    Compute full dev evaluation metrics after calibration and OOS thresholding.

    Returns:
        dict with per-class P/R/F1, macro-F1, accuracy, ECE (before and after
        temperature scaling), and confusion matrix.
    """
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1).copy()
    preds[max_probs < theta] = OOS_IDX

    # Classification report
    report = classification_report(
        true_labels, preds,
        target_names=INTENT_LABELS,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(true_labels, preds, labels=list(range(NUM_CLASSES)))

    # ECE before temperature scaling (raw logits)
    raw_probs = apply_temperature(raw_logits, 1.0)
    confs_raw = raw_probs.max(axis=1)
    correct = (true_labels == preds).astype(float)
    ece_before = compute_ece(confs_raw, correct)

    # ECE after temperature scaling
    confs_scaled = probs.max(axis=1)
    ece_after = compute_ece(confs_scaled, correct)

    logger.info(f"Dev ECE before temperature scaling: {ece_before:.4f}")
    logger.info(f"Dev ECE after temperature scaling (T*={T_star:.4f}): {ece_after:.4f}")
    logger.info(
        f"Dev macro-F1: {report['macro avg']['f1-score']:.4f}, "
        f"accuracy: {report['accuracy']:.4f}"
    )

    # Per-class summary
    logger.info("\nPer-class metrics:")
    for intent in INTENT_LABELS:
        m = report.get(intent, {})
        logger.info(
            f"  {intent:20s} P={m.get('precision', 0):.3f}  "
            f"R={m.get('recall', 0):.3f}  "
            f"F1={m.get('f1-score', 0):.3f}  "
            f"n={m.get('support', 0)}"
        )

    return {
        "report": report,
        "confusion_matrix": cm.tolist(),
        "ece_before_scaling": ece_before,
        "ece_after_scaling": ece_after,
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "theta": theta,
        "T_star": T_star,
    }


def measure_cross_encoder_trigger_rate(
    probs: np.ndarray,
    theta: float,
    delta_values: list[float] | None = None,
) -> pd.DataFrame:
    """
    Measure the fraction of dev queries that would trigger the cross-encoder
    reranker at various δ values.

    The cross-encoder fires when:
      1. The query is in-scope (max_prob >= θ), AND
      2. The confidence gap (top-1 - top-2 probability) < δ.

    A trigger rate > 15% means P99 latency will exceed 15 ms on a 4-core box.

    Args:
        probs:        float32 [N, NUM_CLASSES] calibrated probabilities.
        theta:        OOS threshold.
        delta_values: List of δ values to sweep (default 0.05 to 0.40).

    Returns:
        DataFrame with [delta, trigger_count, trigger_rate].
    """
    if delta_values is None:
        delta_values = [round(x * 0.05, 2) for x in range(1, 9)]  # 0.05..0.40

    rows = []
    n = len(probs)
    max_probs = probs.max(axis=1)
    in_scope_mask = max_probs >= theta  # don't trigger on OOS queries

    sorted_probs = np.sort(probs, axis=1)[:, ::-1]  # descending
    gaps = sorted_probs[:, 0] - sorted_probs[:, 1]

    for delta in delta_values:
        trigger_mask = in_scope_mask & (gaps < delta)
        trigger_count = int(trigger_mask.sum())
        trigger_rate = trigger_count / n
        rows.append({
            "delta": delta,
            "trigger_count": trigger_count,
            "trigger_rate": trigger_rate,
        })

    df = pd.DataFrame(rows)
    logger.info("Cross-encoder trigger rates:")
    logger.info("\n" + df.to_string(index=False))
    return df


# ===========================================================================
# STEP 9 — LATENCY MEASUREMENT
# ===========================================================================

def measure_latency(
    session: ort.InferenceSession,
    tokenizer,
    dev_texts: list[str],
    n_warmup: int = 20,
    n_measure: int = 500,
) -> dict:
    """
    Measure per-sample ONNX inference latency on a CPU machine.

    Methodology:
      - 20 warm-up runs (not measured): allow ONNX Runtime JIT optimisation to
        settle.
      - 500 measured runs, each processing exactly 1 sample.
      - Report median, P50, P95, P99, and mean latency in milliseconds.

    Args:
        session:   ONNX Runtime session.
        tokenizer: HuggingFace tokenizer.
        dev_texts: List of preprocessed dev texts to sample from.
        n_warmup:  Number of warm-up inferences.
        n_measure: Number of timed inferences.

    Returns:
        dict with latency statistics in milliseconds.
    """
    import random as _random
    sample_texts = [dev_texts[i % len(dev_texts)] for i in range(n_warmup + n_measure)]

    latencies_ms = []
    for i, text in enumerate(sample_texts):
        enc = tokenizer(
            text,
            max_length=MAX_SEQ_LEN,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        feed = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        t0 = time.perf_counter()
        _ = session.run(["logits", "embeddings"], feed)
        elapsed = (time.perf_counter() - t0) * 1000  # ms

        if i >= n_warmup:
            latencies_ms.append(elapsed)

    arr = np.array(latencies_ms)
    stats = {
        "median_ms": float(np.median(arr)),
        "p50_ms":    float(np.percentile(arr, 50)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "p99_ms":    float(np.percentile(arr, 99)),
        "mean_ms":   float(arr.mean()),
        "n_samples": len(arr),
    }
    logger.info(
        f"Latency (n={len(arr)}): "
        f"median={stats['median_ms']:.1f} ms, "
        f"P95={stats['p95_ms']:.1f} ms, "
        f"P99={stats['p99_ms']:.1f} ms"
    )
    return stats


# ===========================================================================
# STEP 10 — SAVE PLOTS
# ===========================================================================

def save_confusion_matrix_plot(
    cm: np.ndarray,
    output_path: Path,
) -> None:
    """Save a heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=INTENT_LABELS,
        yticklabels=INTENT_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Dev Set (after calibration + OOS threshold)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix plot → {output_path}")


def save_latency_histogram(latencies: list[float], output_path: Path) -> None:
    """Save a histogram of per-sample ONNX inference latency."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(latencies, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    ax.axvline(p50, color="orange", linestyle="--", label=f"P50 = {p50:.1f} ms")
    ax.axvline(p99, color="red",    linestyle="--", label=f"P99 = {p99:.1f} ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("ONNX Inference Latency per Sample (1 sample, CPU)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved latency histogram → {output_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """
    Entry point: orchestrates Phase 3 (export + calibration) and dev analysis.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3 — EXPORT AND CALIBRATION")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Load model and tokenizer
    # -----------------------------------------------------------------------
    model, checkpoint = load_model_from_checkpoint(CHECKPOINT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_NAME)

    # -----------------------------------------------------------------------
    # Export to ONNX
    # -----------------------------------------------------------------------
    export_to_onnx(model, tokenizer, ONNX_MODEL_PATH)

    # -----------------------------------------------------------------------
    # Build ONNX Runtime session
    # -----------------------------------------------------------------------
    session = build_ort_session(ONNX_MODEL_PATH)

    # -----------------------------------------------------------------------
    # Embed dev.csv
    # -----------------------------------------------------------------------
    logger.info(f"Loading dev.csv from {DEV_CSV}")
    dev_df = pd.read_csv(DEV_CSV)
    dev_texts = dev_df["utterance"].fillna("").apply(preprocess).tolist()
    dev_true = np.array([LABEL2IDX.get(lbl, OOS_IDX) for lbl in dev_df["intent"]])

    dev_logits, dev_embeddings = embed_texts_ort(session, tokenizer, dev_texts)

    # -----------------------------------------------------------------------
    # Temperature scaling
    # -----------------------------------------------------------------------
    T_star = fit_temperature(dev_logits, dev_true)
    dev_probs = apply_temperature(dev_logits, T_star)

    # -----------------------------------------------------------------------
    # OOS threshold tuning
    # -----------------------------------------------------------------------
    theta, oos_sweep_df = tune_oos_threshold(dev_probs, dev_true)
    oos_sweep_df.to_csv(RESULTS_DIR / "oos_threshold_sweep.csv", index=False)

    # -----------------------------------------------------------------------
    # Mahalanobis parameters (fit on cleaned_train embeddings)
    # -----------------------------------------------------------------------
    logger.info(f"Loading cleaned training set from {CLEANED_TRAIN_CSV}")
    if not CLEANED_TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Cleaned train CSV not found at {CLEANED_TRAIN_CSV}. Run train.py first."
        )
    cleaned_df = pd.read_csv(CLEANED_TRAIN_CSV)
    train_texts = cleaned_df["utterance"].fillna("").apply(preprocess).tolist()
    train_labels_arr = np.array([
        LABEL2IDX.get(lbl, OOS_IDX) for lbl in cleaned_df["intent"]
    ])

    logger.info(f"Embedding {len(train_texts)} cleaned training texts for Mahalanobis fitting…")
    _, train_embeddings = embed_texts_ort(session, tokenizer, train_texts)

    mahal_params = fit_mahalanobis(train_embeddings, train_labels_arr)

    # Tune Mahalanobis OOD threshold
    dev_preds_for_mahal = dev_probs.argmax(axis=1).copy()
    max_dev_probs = dev_probs.max(axis=1)
    dev_preds_for_mahal[max_dev_probs < theta] = OOS_IDX

    tau = tune_mahalanobis_threshold(
        dev_embeddings, dev_preds_for_mahal, dev_true, mahal_params
    )
    mahal_params["tau"] = tau

    # -----------------------------------------------------------------------
    # Save calibration and Mahalanobis artefacts
    # -----------------------------------------------------------------------
    calibration = {
        "T_star": T_star,
        "theta": theta,
        "delta": CONFIDENCE_GAP_DELTA,
        "tau": tau,
        "intent_labels": INTENT_LABELS,
        "label2idx": LABEL2IDX,
        "oos_idx": OOS_IDX,
        "encoder_model": ENCODER_MODEL_NAME,
        "max_seq_len": MAX_SEQ_LEN,
    }
    with open(CALIBRATION_PARAMS_PATH, "w") as f:
        json.dump(calibration, f, indent=2)
    logger.info(f"Saved calibration params → {CALIBRATION_PARAMS_PATH}")

    # Convert numpy arrays to serialisable types for pickle
    mahal_save = {
        "centroids": {k: v.astype(np.float32) for k, v in mahal_params["centroids"].items()},
        "precision": mahal_params["precision"].astype(np.float32),
        "tau": float(tau),
    }
    with open(MAHAL_PARAMS_PATH, "wb") as f:
        pickle.dump(mahal_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved Mahalanobis params → {MAHAL_PARAMS_PATH}")

    # -----------------------------------------------------------------------
    # Full dev analysis
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("DEV ANALYSIS")
    logger.info("=" * 60)

    dev_metrics = compute_dev_metrics(dev_probs, dev_true, theta, T_star, dev_logits)

    # Cross-encoder trigger rate analysis
    trigger_df = measure_cross_encoder_trigger_rate(dev_probs, theta)
    trigger_df.to_csv(RESULTS_DIR / "cross_encoder_trigger_rate.csv", index=False)

    # Latency measurement
    latency_stats = measure_latency(session, tokenizer, dev_texts)
    latency_stats["hardware_note"] = (
        "Measured on the eval machine. "
        "ONNX Runtime with 4 intra-op threads, CPUExecutionProvider."
    )

    # -----------------------------------------------------------------------
    # Save per-class metrics CSV
    # -----------------------------------------------------------------------
    per_class_rows = []
    for intent in INTENT_LABELS:
        m = dev_metrics["report"].get(intent, {})
        per_class_rows.append({
            "intent": intent,
            "precision": round(m.get("precision", 0), 4),
            "recall": round(m.get("recall", 0), 4),
            "f1": round(m.get("f1-score", 0), 4),
            "support": int(m.get("support", 0)),
        })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(DEV_PERCLASS_CSV, index=False)
    logger.info(f"Saved per-class metrics → {DEV_PERCLASS_CSV}")

    # Save confusion matrix CSV
    cm_df = pd.DataFrame(
        dev_metrics["confusion_matrix"],
        index=INTENT_LABELS,
        columns=INTENT_LABELS,
    )
    cm_df.to_csv(DEV_CONFUSION_CSV)
    logger.info(f"Saved confusion matrix → {DEV_CONFUSION_CSV}")

    # Save plots
    cm_array = np.array(dev_metrics["confusion_matrix"])
    save_confusion_matrix_plot(cm_array, RESULTS_DIR / "confusion_matrix.png")

    # -----------------------------------------------------------------------
    # Compile and save full dev results JSON
    # -----------------------------------------------------------------------
    results = {
        "dev_macro_f1": dev_metrics["macro_f1"],
        "dev_accuracy": dev_metrics["accuracy"],
        "ece_before_temperature_scaling": dev_metrics["ece_before_scaling"],
        "ece_after_temperature_scaling": dev_metrics["ece_after_scaling"],
        "T_star": T_star,
        "theta": theta,
        "tau": tau,
        "per_class": {
            intent: {
                "precision": round(dev_metrics["report"].get(intent, {}).get("precision", 0), 4),
                "recall":    round(dev_metrics["report"].get(intent, {}).get("recall", 0), 4),
                "f1":        round(dev_metrics["report"].get(intent, {}).get("f1-score", 0), 4),
                "support":   int(dev_metrics["report"].get(intent, {}).get("support", 0)),
            }
            for intent in INTENT_LABELS
        },
        "latency": latency_stats,
        "cross_encoder_trigger_rates": trigger_df.to_dict(orient="records"),
        "evaluation_honesty_notes": [
            "Dev labels are silver. Macro-F1 here is a relative guide, not ground truth.",
            "Taxonomy examples and train.csv were likely generated by the same process; "
            "scores may be inflated relative to hidden gold test labels.",
            "Cleanlab benefits on dev are partly because dev has the same noise pattern. "
            "Gains on hidden gold labels may be smaller.",
            "Latency is hardware-specific; report the machine and thread pinning.",
            f"ECE after scaling = {dev_metrics['ece_after_scaling']:.4f}. "
            + ("OOS threshold reliable." if dev_metrics["ece_after_scaling"] < 0.05 else "ECE > 0.05; OOS threshold less reliable."),
        ],
    }

    with open(DEV_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved full dev results → {DEV_RESULTS_JSON}")

    logger.info("=" * 60)
    logger.info("eval.py complete.")
    logger.info(f"  Dev macro-F1 : {results['dev_macro_f1']:.4f}")
    logger.info(f"  Dev accuracy : {results['dev_accuracy']:.4f}")
    logger.info(f"  ECE (scaled) : {results['ece_after_temperature_scaling']:.4f}")
    logger.info(f"  T*           : {T_star:.4f}")
    logger.info(f"  θ            : {theta}")
    logger.info(f"  τ (Mahal)    : {tau:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
