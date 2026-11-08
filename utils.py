"""
utils.py — QuickCart Intent Pipeline
======================================
Shared utilities used by train.py, eval.py and predict.py:

  - Text preprocessing (Unicode normalisation, whitespace, emoji handling)
  - Seed fixing for reproducibility
  - Logger factory
  - Embedding mean-pooler (pure NumPy, used in eval/predict with ONNX outputs)
  - ECE (Expected Calibration Error) computation
  - Confidence-gap check (triggers cross-encoder)

Nothing here imports torch or transformers so it stays importable in
lightweight inference environments where only ONNX Runtime is installed.
"""

import logging
import re
import unicodedata
import random
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """
    Return a logger that writes to stdout and optionally to a file.

    Args:
        name:     Logger name (usually __name__ of the calling module).
        log_file: Optional Path to a log file; messages are appended.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured; avoid duplicate handlers

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Fix Python, NumPy and (if available) PyTorch seeds.

    Deterministic training is important for a fair before/after comparison:
    if seeds differ between runs, accuracy deltas can be noise rather than
    signal.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic CUDNN ops (small speed cost, full reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed (inference-only environment)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

# Regex patterns compiled once at module load for efficiency.
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_ORDER_ID = re.compile(r"\b[A-Z0-9]{6,}\b")  # e.g. ORD123456


def preprocess(text: str, *, keep_emojis: bool = True) -> str:
    """
    Normalise a raw customer message for embedding.

    Steps applied in order:
      1. Unicode NFKC normalisation — collapses half-width katakana, ligatures,
         etc. to their canonical form.  Important for Hinglish text with mixed
         Unicode code points.
      2. Lowercase.
      3. URL and email removal — these are noise for intent detection.
      4. Strip excess whitespace (tabs, newlines, multiple spaces → single space).
      5. Strip leading/trailing whitespace.

    What we deliberately do NOT do:
      - Remove emojis: emojis carry sentiment and sometimes intent
        (🚚 suggests delivery context; 💳 suggests payment).
        keep_emojis=False is available for ablation experiments.
      - Remove Hinglish words: BGE-small handles romanised Hindi reasonably
        well, and removing them would destroy recall on a significant portion
        of QuickCart's user base.
      - Stemming / lemmatisation: the encoder already maps morphological
        variants to nearby vectors; aggressive stemming introduces noise.

    Args:
        text:         Raw customer message string.
        keep_emojis:  If False, strip all emoji characters.

    Returns:
        Cleaned string, never empty (returns single space if all content stripped).
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Step 1: Unicode NFKC normalisation
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Lowercase
    text = text.lower()

    # Step 3: Remove URLs and email addresses
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)

    # Step 4: Optionally remove emojis
    if not keep_emojis:
        # Remove characters in Unicode emoji blocks
        text = "".join(
            ch for ch in text
            if not unicodedata.category(ch).startswith("So")  # Other Symbol
            and ord(ch) < 0x1F600 or ord(ch) > 0x1F64F  # Emoticons block
        )
        # Simpler fallback using regex for common emoji ranges
        text = re.sub(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
            "\U0001F680-\U0001F6FF"  # Transport and Map
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            " ",
            text,
        )

    # Step 5: Collapse whitespace
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    return text if text else " "


# ---------------------------------------------------------------------------
# Embedding utilities (NumPy, used with ONNX Runtime outputs)
# ---------------------------------------------------------------------------

def mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Compute attention-mask-weighted mean pooling over token embeddings.

    BGE-small outputs one vector per token (shape [B, L, D]).  We aggregate
    to a single sentence vector by averaging over non-padding positions.
    This is the same pooling strategy used in the original BGE paper.

    Args:
        token_embeddings: float32 array of shape [batch, seq_len, hidden_dim].
        attention_mask:   int32 array of shape [batch, seq_len]; 1 = real token,
                          0 = padding.

    Returns:
        float32 array of shape [batch, hidden_dim].
    """
    # Expand mask to match embedding dimension: [B, L] → [B, L, D]
    mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)

    # Zero out padding positions, then sum and normalise
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalise a 2-D array of row vectors in place (new array returned).

    Required before cosine-similarity comparisons and before Mahalanobis
    distance estimation when using tied covariance.

    Args:
        vectors: float32 array of shape [N, D].

    Returns:
        L2-normalised float32 array of same shape.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)  # avoid division by zero
    return vectors / norms


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width bins.

    ECE measures the gap between predicted confidence and empirical accuracy.
    A perfectly calibrated model has ECE = 0.  Post-hoc temperature scaling
    should bring ECE below 0.05 for the OOS threshold to be reliable.

    Args:
        confidences: 1-D float array of predicted max-probability values.
        correct:     1-D bool/int array; 1 if predicted label is correct.
        n_bins:      Number of equal-width probability bins (default 15).

    Returns:
        ECE scalar (lower is better).
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Include the right edge only in the last bin
        if hi < 1.0:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        if mask.sum() == 0:
            continue

        bin_confidence = confidences[mask].mean()
        bin_accuracy = correct[mask].mean()
        bin_weight = mask.sum() / n

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


# ---------------------------------------------------------------------------
# Confidence gap check
# ---------------------------------------------------------------------------

def has_confidence_collision(
    probs: np.ndarray,
    delta: float,
) -> bool:
    """
    Return True if the gap between top-1 and top-2 predicted probabilities
    is smaller than delta — indicating ambiguity that warrants reranking.

    Args:
        probs: 1-D float array of class probabilities (must sum to ~1).
        delta: Minimum gap; collision if gap < delta.

    Returns:
        True → trigger cross-encoder reranker.
        False → return bi-encoder top-1 directly.
    """
    if len(probs) < 2:
        return False
    sorted_probs = np.sort(probs)[::-1]
    gap = float(sorted_probs[0] - sorted_probs[1])
    return gap < delta


# ---------------------------------------------------------------------------
# Mahalanobis distance (NumPy, used at inference)
# ---------------------------------------------------------------------------

def mahalanobis_distance(
    vector: np.ndarray,
    centroid: np.ndarray,
    precision_matrix: np.ndarray,
) -> float:
    """
    Compute Mahalanobis distance between a query vector and a class centroid.

    d(v, μ) = sqrt( (v - μ)ᵀ Σ⁻¹ (v - μ) )

    The precision matrix Σ⁻¹ is the inverse of the shared covariance matrix
    estimated with Ledoit-Wolf shrinkage.  Using a shared (tied) covariance
    instead of per-class covariance avoids singularity in high dimensions.

    Args:
        vector:           1-D float array of shape [D].
        centroid:         1-D float array of shape [D] (class mean embedding).
        precision_matrix: 2-D float array of shape [D, D] (inverse covariance).

    Returns:
        Scalar Mahalanobis distance.
    """
    diff = vector - centroid
    dist_sq = diff @ precision_matrix @ diff
    # Clamp to avoid sqrt of small negatives due to floating-point rounding
    return float(np.sqrt(max(0.0, dist_sq)))
