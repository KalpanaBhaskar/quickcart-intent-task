"""
config.py — QuickCart Intent Pipeline
======================================
Single source of truth for every path, hyperparameter, and constant used
across train.py, eval.py and predict.py.

Keeping configuration here (rather than scattered across files) means:
  - One change propagates everywhere.
  - Hyperparameter search is auditable: edit here, re-run, compare.
  - Scripts stay readable: no magic numbers inside logic.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root directories
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent          # repo root
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SUBMISSIONS_DIR = ROOT / "submissions"
LOGS_DIR = ROOT / "logs"

# Ensure directories exist at import time (safe to call repeatedly)
for _d in (MODELS_DIR, RESULTS_DIR, SUBMISSIONS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------

TAXONOMY_PATH = DATA_DIR / "taxonomy_v1.json"
TRAIN_CSV = DATA_DIR / "train.csv"
DEV_CSV = DATA_DIR / "dev.csv"
TEST_CSV = DATA_DIR / "test.csv"
CLEANED_TRAIN_CSV = DATA_DIR / "cleaned_train.csv"
NOISE_REPORT_CSV = RESULTS_DIR / "noise_report.csv"

# ---------------------------------------------------------------------------
# Model artefacts
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = MODELS_DIR / "best_checkpoint.pt"
ONNX_MODEL_PATH = MODELS_DIR / "model.onnx"
MAHAL_PARAMS_PATH = MODELS_DIR / "mahal_params.pkl"
CALIBRATION_PARAMS_PATH = MODELS_DIR / "calibration_params.json"

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

ENCODER_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE-small-en-v1.5: 33 M params, 384-dim output.
# Chosen because it fits in 15 GB with large headroom, runs ONNX in ~8 ms on
# 4 CPU cores, and already encodes semantic paraphrase relationships that
# TF-IDF cannot capture.

EMBEDDING_DIM = 384          # output dimension of BGE-small
MAX_SEQ_LEN = 128            # tokens; covers >99% of QuickCart messages
                              # (longest real messages ~40 tokens, keeping 128
                              # adds no latency and future-proofs against
                              # verbose customers)

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------

# This list must match the order used when building the linear head.
# Canonical labels, taken from data/taxonomy_v1.json and train.csv.
# out_of_scope must remain the LAST entry (its index is OOS_IDX).
INTENT_LABELS = [
    "order_status",
    "delivery_delay",
    "cancel_order",
    "refund_request",
    "missing_item",
    "wrong_item",
    "payment_failed",
    "promo_code_issue",
    "change_address",
    "account_help",
    "talk_to_agent",
    "out_of_scope",
]
NUM_CLASSES = len(INTENT_LABELS)        # 12
LABEL2IDX = {lbl: i for i, lbl in enumerate(INTENT_LABELS)}
IDX2LABEL = {i: lbl for lbl, i in LABEL2IDX.items()}
OOS_IDX = LABEL2IDX["out_of_scope"]     # 11

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

SEED = 42

# Joint loss weights: λ₁ · L_InfoNCE + λ₂ · L_CE
# Rationale: CE dominates (0.7) for explicit boundary learning; InfoNCE (0.3)
# regularises the embedding geometry so close-intent pairs (refund vs wrong_item)
# don't collapse.
LAMBDA_INFONCE = 0.3
LAMBDA_CE = 0.7

# InfoNCE temperature τ (not the same as calibration temperature T*).
# Smaller τ → sharper contrast between positives and negatives in embedding
# space.  0.07 is the standard value from SimCSE / BGE papers.
INFONCE_TEMP = 0.07

BATCH_SIZE = 64              # per-GPU or per-CPU; adjust down to 32 if OOM
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5         # standard for fine-tuning sentence encoders
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1           # fraction of total steps used for LR warm-up

# Taxonomy anchor weight: each taxonomy example is repeated 5× in the
# training dataset so silver noise cannot drag centroids away from the
# human-verified definition of each intent.
TAXONOMY_ANCHOR_WEIGHT = 5

# Cleanlab cross-validation folds.  5-fold is the standard; more folds =
# better probability estimates but longer runtime.
CLEANLAB_CV_FOLDS = 5

# Early stopping: halt training if dev Macro-F1 does not improve for
# PATIENCE consecutive epochs; restore the best checkpoint.
PATIENCE = 3

# LR reduction: halve LR if dev Macro-F1 does not improve for
# LR_PATIENCE consecutive epochs.
LR_PATIENCE = 2
LR_FACTOR = 0.5

# ---------------------------------------------------------------------------
# Calibration and OOS/OOD thresholds
# ---------------------------------------------------------------------------

# θ sweep range for OOS threshold tuning (swept on dev.csv).
# θ is the minimum max-probability; queries below θ → out_of_scope.
OOS_THETA_SWEEP = [round(x * 0.05, 2) for x in range(4, 17)]  # 0.20 .. 0.80

# δ: confidence gap below which the cross-encoder reranker is triggered.
# Tune on dev to keep trigger rate < 15%.
CONFIDENCE_GAP_DELTA = 0.15

# τ: Mahalanobis distance above which a query is flagged as OOD (but still
# returns a best-effort label; routing to human review is a downstream choice).
# Initial value; tuned on dev to keep OOD flag rate reasonable (~5–10%).
MAHAL_OOD_TAU = 4.0

# ---------------------------------------------------------------------------
# Cross-encoder (reranker)
# ---------------------------------------------------------------------------

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# 22 M params; runs in ~30 ms on CPU.  Triggered only on confidence collisions
# (~15% of queries) so median latency stays under 15 ms.

CROSS_ENCODER_TOP_K = 3   # score only the top-3 bi-encoder candidates

# ---------------------------------------------------------------------------
# ONNX export settings
# ---------------------------------------------------------------------------

ONNX_OPSET = 17
# Opset 17 supports all operators used by BGE-small (attention, layer norm,
# GELU).  Dynamic sequence-length axis is set at export time.

# ---------------------------------------------------------------------------
# Evaluation output paths
# ---------------------------------------------------------------------------

DEV_RESULTS_JSON = RESULTS_DIR / "eval_dev.json"
DEV_CONFUSION_CSV = RESULTS_DIR / "confusion_matrix.csv"
DEV_PERCLASS_CSV = RESULTS_DIR / "per_class_metrics.csv"
DEV_LATENCY_CSV = RESULTS_DIR / "latency_histogram.csv"

# Submission filename pattern — replace with your name before submitting
SUBMITTER_NAME = "Kalpana"
SUBMISSION_CSV = SUBMISSIONS_DIR / f"{SUBMITTER_NAME}_predictions.csv"
