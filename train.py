"""
train.py — QuickCart Intent Pipeline  |  Phase 1 (Data Engine) + Phase 2 (Training)
======================================================================================

What this file does
-------------------
1. Load taxonomy_v1.json.  Inject canonical examples for the two zero-example
   intents (missing_item, wrong_item) based on the hidden-label rules.
2. Load train.csv.  Run Cleanlab Confident Learning to detect and remove
   silver-label noise.  Save the cleaned set and a noise report.
3. Build a PyTorch Dataset:
     - Each taxonomy example is replicated TAXONOMY_ANCHOR_WEIGHT (5×) times.
     - Cleaned train rows are added once each.
   This ensures human-verified taxonomy centroid geometry is preserved even
   after the linear head starts pushing class boundaries.
4. Instantiate BAAI/bge-small-en-v1.5 encoder + a linear classification head.
5. Train with a joint loss:
     L_total = λ₁ · L_InfoNCE  +  λ₂ · L_CE
   where L_InfoNCE uses multi-positive pairs within each batch to build a clean
   embedding geometry, and L_CE draws explicit hyperplane boundaries.
6. After each epoch: evaluate Macro-F1 and ECE on dev.csv.
   Apply adaptive LR (halve if no improvement for LR_PATIENCE epochs) and
   early stopping (halt + restore if no improvement for PATIENCE epochs).
7. Save best_checkpoint.pt.

Run
---
  python -m train

Dependencies (all in requirements.txt):
  torch, transformers, cleanlab, scikit-learn, pandas, numpy, tqdm, accelerate
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from cleanlab.filter import find_label_issues

from config import (
    TAXONOMY_PATH, TRAIN_CSV, DEV_CSV,
    CLEANED_TRAIN_CSV, NOISE_REPORT_CSV,
    CHECKPOINT_PATH, LOGS_DIR,
    ENCODER_MODEL_NAME, EMBEDDING_DIM, MAX_SEQ_LEN,
    INTENT_LABELS, NUM_CLASSES, LABEL2IDX, IDX2LABEL, OOS_IDX,
    SEED, LAMBDA_INFONCE, LAMBDA_CE, INFONCE_TEMP,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO,
    TAXONOMY_ANCHOR_WEIGHT, CLEANLAB_CV_FOLDS,
    PATIENCE, LR_PATIENCE, LR_FACTOR,
)
from utils import get_logger, set_seed, preprocess, compute_ece

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logger = get_logger(__name__, LOGS_DIR / "train.log")
set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")


# ===========================================================================
# PHASE 1 — DATA ENGINE
# ===========================================================================

# ---------------------------------------------------------------------------
# 1a. Taxonomy loading and zero-example intent injection
# ---------------------------------------------------------------------------

def load_taxonomy(path: Path) -> dict[str, list[str]]:
    """
    Read data/taxonomy_v1.json, format:
        {"intents": [{"name": str, "examples": [str, ...]}, ...]}
    Returns {intent_name: [examples]}. Raises on malformed input.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or "intents" not in raw:
        raise ValueError(f"{path}: expected top-level key 'intents'")

    taxonomy: dict[str, list[str]] = {}
    for item in raw["intents"]:
        name = item["name"]
        if name in taxonomy:
            raise ValueError(f"Duplicate intent in taxonomy: {name}")
        taxonomy[name] = list(item.get("examples", []))
    return taxonomy


def validate_taxonomy(taxonomy: dict[str, list[str]]) -> None:
    """
    Hard checks: every label in INTENT_LABELS must exist in the taxonomy
    with at least one example, and the taxonomy must not contain unknown names.
    """
    unknown = set(taxonomy) - set(INTENT_LABELS)
    missing = set(INTENT_LABELS) - set(taxonomy)
    empty = [k for k in INTENT_LABELS if k in taxonomy and not taxonomy[k]]

    problems = []
    if unknown: problems.append(f"unknown intents in taxonomy: {sorted(unknown)}")
    if missing: problems.append(f"intents missing from taxonomy: {sorted(missing)}")
    if empty:   problems.append(f"intents with zero examples: {empty}")
    if problems:
        raise ValueError("Taxonomy validation failed: " + "; ".join(problems))

    for intent in INTENT_LABELS:
        logger.info(f"  Taxonomy examples — {intent}: {len(taxonomy[intent])}")


# ---------------------------------------------------------------------------
# 1b. Cleanlab noise detection and removal
# ---------------------------------------------------------------------------

def run_cleanlab(
    df: pd.DataFrame,
    cv_folds: int = CLEANLAB_CV_FOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify and remove mislabelled rows from train.csv using Confident Learning.

    How it works:
      1. Use StratifiedKFold cross-validation to get out-of-fold predicted
         probabilities from a lightweight LogisticRegression on TF-IDF features.
         (We use TF-IDF here — not the neural encoder — because Cleanlab only
          needs rough probability estimates, not perfect ones, and running the
          full encoder on 30k samples in CV is slow.)
      2. Feed those probabilities + the silver labels to Cleanlab's
         find_label_issues(), which implements the Confident Learning algorithm.
      3. Remove flagged rows from the training set.
      4. Return the cleaned DataFrame and a noise report DataFrame.

    Args:
        df:        train.csv loaded as DataFrame with columns [intent, utterance].
        cv_folds:  Number of cross-validation folds (default 5).

    Returns:
        (cleaned_df, noise_report_df)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    logger.info(f"Running Cleanlab on {len(df)} training rows with {cv_folds}-fold CV…")
    t0 = time.time()

    # --- Encode labels as integers ---
    # Only keep rows whose label is in our known intent set
    valid_mask = df["intent"].isin(LABEL2IDX)
    if not valid_mask.all():
        bad = df.loc[~valid_mask, "intent"].value_counts()
        raise ValueError(f"train.csv has labels not in INTENT_LABELS:\n{bad}")
    
    # df = df[valid_mask].copy().reset_index(drop=True)

    y = df["intent"].map(LABEL2IDX).values.astype(int)

    # --- TF-IDF features (fast, good enough for noise detection) ---
    texts = df["utterance"].fillna("").apply(preprocess).tolist()
    tfidf = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True)
    X = tfidf.fit_transform(texts)

    # --- Cross-validated probability estimates ---
    # LogisticRegression with multi_class='multinomial' gives calibrated
    # probabilities per class, which is what Cleanlab needs.
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
        random_state=SEED,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

    # cross_val_predict with method='predict_proba' gives OOF probabilities
    # shape: [N, num_classes]
    prob_pred = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")

    # Ensure the probability matrix has columns for every class index
    # (cross_val_predict only returns columns present in y; fill missing ones)
    all_classes = sorted(set(y))
    if len(all_classes) < NUM_CLASSES:
        # Build a full matrix with zeros for absent classes
        full_probs = np.zeros((len(df), NUM_CLASSES), dtype=np.float64)
        for col_i, class_idx in enumerate(all_classes):
            full_probs[:, class_idx] = prob_pred[:, col_i]
        prob_pred = full_probs

    # --- Find label issues ---
    issue_indices = find_label_issues(
        labels=y,
        pred_probs=prob_pred,
        return_indices_ranked_by="self_confidence",
    )
    # Cleanlab returns ranked integer indices when return_indices_ranked_by is set.
    label_issues = np.zeros(len(df), dtype=bool)
    label_issues[issue_indices] = True

    n_flagged = label_issues.sum()
    logger.info(
        f"Cleanlab flagged {n_flagged} / {len(df)} rows "
        f"({100 * n_flagged / len(df):.1f}%) as likely mislabelled"
    )

    # --- Build noise report ---
    noise_df = df[label_issues].copy()
    noise_df["cleanlab_flag"] = True
    # Add the most likely correct label (argmax of predicted probabilities)
    noise_df["predicted_intent"] = prob_pred[label_issues].argmax(axis=1)
    noise_df["predicted_intent"] = noise_df["predicted_intent"].map(IDX2LABEL)

    per_class_noise = (
        noise_df.groupby("intent")
        .size()
        .reset_index(name="n_flagged")
    )
    per_class_total = df.groupby("intent").size().reset_index(name="n_total")
    noise_report = per_class_total.merge(per_class_noise, on="intent", how="left")
    noise_report["n_flagged"] = noise_report["n_flagged"].fillna(0).astype(int)
    noise_report["noise_rate"] = noise_report["n_flagged"] / noise_report["n_total"]
    noise_report = noise_report.sort_values("noise_rate", ascending=False)

    logger.info("Per-class noise rates:\n" + noise_report.to_string(index=False))

    # Save top-10 most ambiguous removed samples for manual audit
    top10 = noise_df.head(10)
    logger.info("Top-10 most ambiguous removed samples:")
    for _, row in top10.iterrows():
        logger.info(
            f"  [{row['intent']} → {row['predicted_intent']}] "
            f"{row['utterance'][:80]}"
        )

    # --- Remove flagged rows from training set ---
    cleaned_df = df[~label_issues].reset_index(drop=True)
    logger.info(
        f"Cleaned training set: {len(cleaned_df)} rows "
        f"(removed {len(df) - len(cleaned_df)}). "
        f"Elapsed: {time.time() - t0:.1f}s"
    )

    return cleaned_df, noise_report


# ---------------------------------------------------------------------------
# 1c. Build the full training corpus with taxonomy anchors
# ---------------------------------------------------------------------------

def build_corpus(
    taxonomy: dict[str, list[str]],
    cleaned_train: pd.DataFrame,
) -> list[tuple[str, int]]:
    """
    Combine taxonomy anchor examples (weighted at 5×) with the cleaned
    training set.

    Why weight taxonomy examples?
    The taxonomy is the only human-verified signal.  Silver labels in
    train.csv are noisy.  Repeating taxonomy examples TAXONOMY_ANCHOR_WEIGHT
    times ensures centroid geometry cannot drift far from the true intent
    definition even if many training rows are borderline or subtly wrong.

    Args:
        taxonomy:      {intent: [example_strings]} from load_and_expand_taxonomy.
        cleaned_train: Cleaned train.csv with columns [intent, utterance].

    Returns:
        List of (preprocessed_utterance, label_index) tuples.
    """
    corpus: list[tuple[str, int]] = []

    # --- Taxonomy anchors at 5× weight ---
    for intent, examples in taxonomy.items():
        idx = LABEL2IDX.get(intent)
        if idx is None:
            logger.warning(f"Taxonomy intent '{intent}' not in LABEL2IDX; skipping")
            continue
        for ex in examples:
            text = preprocess(ex)
            for _ in range(TAXONOMY_ANCHOR_WEIGHT):
                corpus.append((text, idx))

    # --- Cleaned training set at 1× weight ---
    for _, row in cleaned_train.iterrows():
        idx = LABEL2IDX.get(row["intent"])
        if idx is None:
            continue
        text = preprocess(str(row["utterance"]))
        corpus.append((text, idx))

    # Shuffle so taxonomy anchors and train rows are intermixed across batches
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(corpus))
    corpus = [corpus[i] for i in indices]

    # Log class distribution
    from collections import Counter
    counts = Counter(idx for _, idx in corpus)
    logger.info("Training corpus class distribution:")
    for idx, lbl in IDX2LABEL.items():
        logger.info(f"  {lbl}: {counts.get(idx, 0)}")
    logger.info(f"Total training items: {len(corpus)}")

    return corpus


# ===========================================================================
# PHASE 2 — MODEL
# ===========================================================================

# ---------------------------------------------------------------------------
# 2a. Dataset
# ---------------------------------------------------------------------------

class IntentDataset(Dataset):
    """
    PyTorch Dataset wrapping (text, label) pairs.

    Tokenisation is deferred to __getitem__ (lazy) rather than pre-computed
    at construction time, which keeps RAM usage reasonable for 30k+ samples
    when tokens are stored as tensors.

    Args:
        corpus:    List of (preprocessed_text, label_index) tuples.
        tokenizer: HuggingFace tokenizer instance.
        max_len:   Maximum token length.
    """

    def __init__(
        self,
        corpus: list[tuple[str, int]],
        tokenizer,
        max_len: int = MAX_SEQ_LEN,
    ) -> None:
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx: int) -> dict:
        text, label = self.corpus[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",   # pad to max_len for batching efficiency
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),        # [L]
            "attention_mask": enc["attention_mask"].squeeze(0),  # [L]
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 2b. Balanced batch sampler
# ---------------------------------------------------------------------------

def build_balanced_sampler(labels: list[int]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that draws an equal expected number of
    samples from each class per epoch.

    Without balancing, overrepresented classes (e.g. order_status) dominate
    gradient updates and underrepresented ones (e.g. promo_code_issue) get poor
    boundary coverage.

    Args:
        labels: List of integer class indices for every item in the dataset.

    Returns:
        WeightedRandomSampler instance to pass to DataLoader.
    """
    from collections import Counter
    counts = Counter(labels)
    # Weight for each sample = 1 / class_frequency
    weights = [1.0 / counts[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# 2c. The model: BGE-small encoder + linear head
# ---------------------------------------------------------------------------

class IntentClassifier(nn.Module):
    """
    Sentence encoder (BGE-small-en-v1.5) with an attached linear classification
    head.

    Architecture:
      Input tokens → BGE-small transformer → mean pool → 384-dim vector
      → Linear(384, NUM_CLASSES) → 13 raw logits

    The mean pooling is done inside the forward pass so the entire graph
    (encoder + head) can be exported to ONNX in one shot with dynamic sequence
    length.

    Why mean pool instead of [CLS] token?
    BGE is trained with mean pooling as the sentence representation strategy.
    Using [CLS] would require a full re-training cycle to work well.
    """

    def __init__(self, model_name: str = ENCODER_MODEL_NAME) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # Freeze the bottom 6 of 12 transformer layers to reduce overfitting
        # on the relatively small QuickCart training set.  The top 6 layers
        # (and the head) are trained freely.
        for layer in self.encoder.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False
        # Classification head
        self.head = nn.Linear(EMBEDDING_DIM, NUM_CLASSES)

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      Token IDs, shape [B, L].
            attention_mask: Mask, shape [B, L].

        Returns:
            (logits, embeddings)
              logits:     [B, NUM_CLASSES] raw (pre-softmax) class scores.
              embeddings: [B, EMBEDDING_DIM] L2-normalised sentence vectors.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # outputs.last_hidden_state: [B, L, D]
        token_embs = outputs.last_hidden_state

        # Mean pool over non-padding positions
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        summed = (token_embs * mask_expanded).sum(dim=1)      # [B, D]
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)     # [B, 1]
        embeddings = summed / counts                          # [B, D]

        # L2 normalise (BGE is trained with normalised embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=-1)     # [B, D]

        logits = self.head(embeddings)                        # [B, NUM_CLASSES]
        return logits, embeddings


# ---------------------------------------------------------------------------
# 2d. Loss functions
# ---------------------------------------------------------------------------

def infonce_loss(
    embeddings: torch.Tensor,  # [B, D]
    labels: torch.Tensor,      # [B]
    temperature: float = INFONCE_TEMP,
) -> torch.Tensor:
    """
    Multi-Positive InfoNCE (Contrastive) Loss.

    For each anchor in the batch, all other samples sharing the same label are
    positives; all samples with different labels are negatives.  The loss
    minimises:

        L = -1/N Σ_i  log(  Σ_{j: y_j = y_i, j≠i} exp(sim(i,j)/τ)
                           / Σ_{j: j≠i} exp(sim(i,j)/τ)  )

    This pulls same-intent embeddings together and pushes different-intent
    embeddings apart in the 384-dim space.

    Args:
        embeddings: L2-normalised sentence vectors, shape [B, D].
        labels:     Integer class labels, shape [B].
        temperature: τ; smaller = sharper contrast.

    Returns:
        Scalar loss tensor.
    """
    B = embeddings.shape[0]
    # Pairwise cosine similarity matrix: [B, B]
    # (embeddings are already L2-normalised, so dot product = cosine similarity)
    sim_matrix = embeddings @ embeddings.T / temperature  # [B, B]

    # Build positive/negative masks
    labels_row = labels.unsqueeze(1)  # [B, 1]
    labels_col = labels.unsqueeze(0)  # [1, B]
    pos_mask = (labels_row == labels_col)  # [B, B]; True where same class

    # Exclude self-pairs from both numerator and denominator
    eye = torch.eye(B, device=embeddings.device, dtype=torch.bool)
    pos_mask = pos_mask & ~eye

    # If a sample has no positives in the batch (can happen with heavy
    # augmentation), skip it to avoid NaN
    has_positive = pos_mask.any(dim=1)  # [B]
    if not has_positive.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # For numerical stability, subtract row max before exp (log-sum-exp trick)
    sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values.detach()

    exp_sim = torch.exp(sim_matrix)                    # [B, B]
    # Zero out self-pairs in denominator
    exp_sim_no_self = exp_sim * (~eye).float()
    denom = exp_sim_no_self.sum(dim=1)                 # [B]

    # Numerator: sum of exp(sim) over positives only
    numer = (exp_sim * pos_mask.float()).sum(dim=1)    # [B]

    # Safe log: avoid log(0) if numer is 0 for samples without positives
    numer = numer.clamp(min=1e-9)
    denom = denom.clamp(min=1e-9)
    loss_per_sample = -torch.log(numer / denom)       # [B]

    # Average only over samples that had at least one positive
    return loss_per_sample[has_positive].mean()


# ---------------------------------------------------------------------------
# 2e. Dev evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_on_dev(
    model: IntentClassifier,
    tokenizer,
    dev_df: pd.DataFrame,
) -> dict:
    """
    Run the model on dev.csv and return Macro-F1 and ECE.

    Dev labels are silver, so this is a relative (not absolute) guide.

    Args:
        model:    IntentClassifier in eval mode.
        tokenizer: HuggingFace tokenizer.
        dev_df:   dev.csv DataFrame with columns [intent, utterance].

    Returns:
        dict with keys: macro_f1, ece, accuracy
    """
    model.eval()

    all_preds, all_labels, all_confs = [], [], []
    batch_size = 128  # larger batch for fast evaluation

    texts = dev_df["utterance"].fillna("").apply(preprocess).tolist()
    true_labels = [LABEL2IDX.get(lbl, OOS_IDX) for lbl in dev_df["intent"]]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        enc = tokenizer(
            batch_texts,
            max_length=MAX_SEQ_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits, _ = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)

        all_preds.extend(preds.tolist())
        all_confs.extend(confs.tolist())

    all_labels_arr = np.array(true_labels)
    all_preds_arr = np.array(all_preds)
    all_confs_arr = np.array(all_confs)

    macro_f1 = f1_score(all_labels_arr, all_preds_arr, average="macro", zero_division=0)
    accuracy = (all_labels_arr == all_preds_arr).mean()
    correct = (all_labels_arr == all_preds_arr).astype(float)
    ece = compute_ece(all_confs_arr, correct)

    return {"macro_f1": macro_f1, "accuracy": accuracy, "ece": ece}


# ===========================================================================
# MAIN TRAINING LOOP
# ===========================================================================

def main() -> None:
    """
    Entry point: orchestrates Phases 1 and 2 end-to-end.
    """
    # -----------------------------------------------------------------------
    # Phase 1a: Load and expand taxonomy
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 1 — DATA ENGINE")
    logger.info("=" * 60)

    taxonomy = load_taxonomy(TAXONOMY_PATH)
    validate_taxonomy(taxonomy)

    # -----------------------------------------------------------------------
    # Phase 1b: Load train.csv, run Cleanlab
    # -----------------------------------------------------------------------
    logger.info(f"Loading training data from {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    logger.info(f"Raw training rows: {len(train_df)}")

    # Basic sanity check
    if "intent" not in train_df.columns or "utterance" not in train_df.columns:
        raise ValueError(
            f"train.csv must have 'intent' and 'utterance' columns. "
            f"Found: {list(train_df.columns)}"
        )

    cleaned_train, noise_report = run_cleanlab(train_df)

    # Save cleaned set and noise report
    cleaned_train.to_csv(CLEANED_TRAIN_CSV, index=False)
    logger.info(f"Saved cleaned training set → {CLEANED_TRAIN_CSV}")

    noise_report.to_csv(NOISE_REPORT_CSV, index=False)
    logger.info(f"Saved noise report → {NOISE_REPORT_CSV}")

    # -----------------------------------------------------------------------
    # Phase 1c: Build corpus with taxonomy anchors
    # -----------------------------------------------------------------------
    corpus = build_corpus(taxonomy, cleaned_train)

    # -----------------------------------------------------------------------
    # Phase 2: Load tokenizer and model
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 2 — TRAINING")
    logger.info("=" * 60)

    logger.info(f"Loading tokenizer and model: {ENCODER_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_NAME)
    model = IntentClassifier(ENCODER_MODEL_NAME).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: {n_params:,} total, {n_trainable:,} trainable"
    )

    # -----------------------------------------------------------------------
    # Build dataset and dataloader
    # -----------------------------------------------------------------------
    dataset = IntentDataset(corpus, tokenizer)
    labels_list = [lbl for _, lbl in corpus]
    sampler = build_balanced_sampler(labels_list)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,  # 0 for compatibility; increase if CPU is not bottleneck
        pin_memory=(DEVICE.type == "cuda"),
    )

    # -----------------------------------------------------------------------
    # Load dev.csv for epoch-level evaluation
    # -----------------------------------------------------------------------
    logger.info(f"Loading dev data from {DEV_CSV}")
    dev_df = pd.read_csv(DEV_CSV)
    logger.info(f"Dev rows: {len(dev_df)}")

    # -----------------------------------------------------------------------
    # Optimiser and scheduler
    # -----------------------------------------------------------------------
    # Separate learning rates: smaller LR for the pre-trained encoder layers,
    # larger LR for the randomly-initialised classification head.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": LEARNING_RATE},
            {"params": model.head.parameters(), "lr": LEARNING_RATE * 10},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    # Linear warmup then linear decay schedule
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    # label_smoothing=0.05 reduces overconfidence on silver labels

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_f1 = 0.0
    no_improve_count = 0     # for early stopping
    lr_no_improve = 0        # for adaptive LR

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = total_ce = total_nce = 0.0
        t_epoch = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits, embeddings = model(input_ids, attention_mask)

            # Joint loss
            l_ce = ce_loss_fn(logits, labels)
            l_nce = infonce_loss(embeddings, labels)
            loss = LAMBDA_CE * l_ce + LAMBDA_INFONCE * l_nce

            loss.backward()
            # Gradient clipping prevents exploding gradients during early warmup
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_ce += l_ce.item()
            total_nce += l_nce.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{l_ce.item():.4f}",
                nce=f"{l_nce.item():.4f}",
            )

        avg_loss = total_loss / len(loader)
        avg_ce = total_ce / len(loader)
        avg_nce = total_nce / len(loader)
        epoch_time = time.time() - t_epoch

        # --- Dev evaluation ---
        dev_metrics = evaluate_on_dev(model, tokenizer, dev_df)
        macro_f1 = dev_metrics["macro_f1"]
        ece = dev_metrics["ece"]
        accuracy = dev_metrics["accuracy"]

        logger.info(
            f"Epoch {epoch:02d} | "
            f"loss={avg_loss:.4f} (CE={avg_ce:.4f}, NCE={avg_nce:.4f}) | "
            f"Dev: macro_F1={macro_f1:.4f}, accuracy={accuracy:.4f}, ECE={ece:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # --- Checkpoint: save if best dev Macro-F1 ---
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "macro_f1": macro_f1,
                    "ece": ece,
                    "config": {
                        "encoder_model": ENCODER_MODEL_NAME,
                        "num_classes": NUM_CLASSES,
                        "intent_labels": INTENT_LABELS,
                    },
                },
                CHECKPOINT_PATH,
            )
            logger.info(f"  → Saved best checkpoint (macro_F1={macro_f1:.4f})")
            no_improve_count = 0
            lr_no_improve = 0
        else:
            no_improve_count += 1
            lr_no_improve += 1
            logger.info(
                f"  → No improvement. "
                f"Early-stop counter: {no_improve_count}/{PATIENCE}, "
                f"LR counter: {lr_no_improve}/{LR_PATIENCE}"
            )

            # Adaptive LR: halve if no improvement for LR_PATIENCE epochs
            if lr_no_improve >= LR_PATIENCE:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= LR_FACTOR
                logger.info(
                    f"  → Reduced LR by {LR_FACTOR}×. "
                    f"New LRs: {[pg['lr'] for pg in optimizer.param_groups]}"
                )
                lr_no_improve = 0

            # Early stopping
            if no_improve_count >= PATIENCE:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs. "
                    f"Best dev Macro-F1: {best_f1:.4f}"
                )
                break

    logger.info(f"Training complete. Best checkpoint: {CHECKPOINT_PATH}")
    logger.info(f"Best dev Macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
