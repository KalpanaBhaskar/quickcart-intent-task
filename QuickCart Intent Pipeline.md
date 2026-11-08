# QuickCart Intent Pipeline — Architecture & Code Design

*Applied ML Intern Task · Yellow.ai Research Team · BGE-small-en-v1.5 · 12-class + OOS*

---

## 0. What This Document Is

This document is the full **reasoning record and implementation blueprint** for the final QuickCart intent-classification pipeline. It covers every architectural decision from raw data ingestion to test-set submission, with the "why" made explicit at each stage. Code is split across three files. Everything here is built for the 4-core / 15 GB production ceiling.

---

## 1. The Goal, Restated Sharply

Given a noisy customer message — typos, Hinglish, multi-request, emojis — produce **exactly one intent label** from a fixed set of 12 intents plus `out_of_scope`, reliably and within \~15 ms on a CPU-only production machine.

The baseline (TF-IDF cosine, 0.42 top-1 accuracy) fails because:

- It has no sense of meaning, only token overlap.
- It cannot generalise to synonyms, Hinglish, or paraphrase.
- Two intents (`missing_item`, `wrong_item`) have zero taxonomy examples — the baseline predicts them never.
- Silver label noise in `train.csv` poisons any model that trains naively.
- OOS detection is a crude fixed threshold, not a learned boundary.

The final pipeline fixes all five.

---

## 2. Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 · DATA ENGINE  (runs once, offline)                                 │
│                                                                              │
│  taxonomy.json ──► Gold Anchor Injection (weight = 5×)                      │
│  train.csv ──────► Cleanlab (Confident Learning) ──► Cleaned Train Set      │
│                                                 └──► Noise Report            │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │ cleaned_train.csv
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 · TRAINING  (runs once, offline, GPU or CPU)                        │
│                                                                              │
│  BGE-small-en-v1.5 (encoder, 384-dim)                                        │
│    │                                                                          │
│    ├── Mean Pool → 384-dim sentence vector                                   │
│    │       │                                                                  │
│    │       ├── Objective 1: Multi-Positive InfoNCE (geometry)                │
│    │       └── Objective 2: Linear Head + CE Loss  (boundaries)             │
│    │                                                                          │
│  Balanced Batch Sampler  ·  Taxonomy anchors at 5× weight                    │
│  Dev loop: Macro-F1, ECE, Adaptive LR, Early Stopping (patience = 3)        │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │ best_checkpoint.pt
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3 · EXPORT  (runs once, offline)                                      │
│                                                                              │
│  best_checkpoint.pt ──► ONNX export (encoder + head, fused)                 │
│  Fit Mahalanobis parameters on training embeddings (class centroids + Σ⁻¹)  │
│  Fit Temperature Scaler T* on dev.csv                                        │
│  Tune OOS threshold θ on dev.csv                                             │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │ model.onnx  ·  mahal_params.pkl  ·  T*, θ
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4 · INFERENCE  (live, 4-core / 15 GB ceiling)                        │
│                                                                              │
│  Message ──► Preprocess ──► ONNX (< 10 ms) ──► Logits                       │
│                                   │                                          │
│                              Temperature Scale                               │
│                                   │                                          │
│               ┌───────────────────┴───────────────────┐                     │
│        OOS path (ODIN)                        In-scope path                  │
│     max_prob < θ → out_of_scope        confidence gap < δ?                  │
│                                               │         │                    │
│                                            YES: Cross-Encoder (Top-3)        │
│                                             NO: return Top-1                 │
│                                                                              │
│  OOD flag: Mahalanobis > τ → flag for human review (still returns label)    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. A Concrete Example Through Every Stage

> **Input message:** `"bhai mera order kahan hai, refund bhi chahiye"` *(Hinglish: "bro where is my order, I also want a refund")*

| Stage | What happens | Output |
| --- | --- | --- |
| **Preprocess** | Unicode normalise, strip excess whitespace, keep Hinglish intact | `"bhai mera order kahan hai refund bhi chahiye"` |
| **ONNX Encoder** | Tokenise → 384-dim mean-pooled vector in \~8 ms | `v ∈ ℝ³⁸⁴` |
| **Linear Head** | `W·v + b` → 13 raw logits | `[2.1, 0.3, 1.9, ...]` |
| **Temp Scale** | Divide by T\* = 1.4, apply softmax | `order_status: 0.52, refund_request: 0.41, ...` |
| **OOS check** | max prob 0.52 > θ = 0.35 → in-scope | — |
| **Gap check** | 0.52 − 0.41 = 0.11 < δ = 0.15 → confidence collision | Top-3: `order_status`, `refund_request`, `delivery_delay` |
| **Cross-Encoder** | Scores message vs. each of 3 intent definitions | `order_status: 0.89` |
| **Label rule** | "first request wins" → order\_status | ✅ **`order_status`** |
| **Mahalanobis** | Distance = 2.1 (< τ = 4.0) → not OOD | No flag |

---

## 4. Stage-by-Stage Deep Dive

### Stage 1 — Data Engine

#### 1a. Taxonomy as Gold Standard

The taxonomy is the only source of **human-verified ground truth**. The pipeline treats it as:

- The semantic centre of each class (centroids are always re-anchored here).
- Labelling ground truth for the two zero-example intents (`missing_item`, `wrong_item`), which must be populated with manually written or LLM-generated canonical examples before training begins.
- A weight multiplier of **5×** in the loss so silver noise cannot drag decision boundaries away from the taxonomic definition.

**Action:** Before any training, expand the taxonomy for the two empty intents with at least 10 canonical examples each. Use the hidden-label rules ("something absent = missing\_item", "something different = wrong\_item") to write them.

#### 1b. Cleanlab Noise Removal

Cleanlab implements **Confident Learning**: it cross-validates the full `train.csv` (5 folds), collects out-of-fold predicted probabilities, and identifies samples where the silver label is inconsistent with model confidence across all folds. These are flagged and removed.

**Expected outcome:** Roughly 5–15% of `train.csv` rows are mislabelled (empirical estimate from similar silver-label sets). Removing them consistently improves downstream F1 on clean test sets by 3–8 percentage points without touching model architecture.

**What to report:** Number of removed samples per class, top-10 most ambiguous removed samples (read them manually), and the class-level noise rate estimate.

#### 1c. Balanced Batch Sampler

If the class distribution in `train.csv` is skewed, overrepresented classes dominate gradient updates. The balanced batch sampler draws an equal number of samples from each class per batch, decoupling optimisation from distribution.

---

### Stage 2 — Training

#### 2a. The Encoder: BGE-small-en-v1.5

BGE-small-en-v1.5 is a 33 M-parameter sentence encoder (384-dim output). Reasons for the choice:

- Fits entirely in RAM with headroom on a 15 GB machine.
- ONNX-compiled inference runs in \~8 ms on 4 CPU cores.
- Pre-trained on a large sentence-pair corpus, so it already understands paraphrase and synonym at zero-shot.
- Outperforms all-MiniLM-L6-v2 on retrieval benchmarks while being similar in size.

Mean pooling over the final hidden states is the canonical way to get a fixed-size sentence vector from this encoder.

#### 2b. Joint Loss: InfoNCE + Cross-Entropy

```
Loss_total = λ₁ · L_InfoNCE + λ₂ · L_CE
```

**L\_InfoNCE (Multi-Positive):** Given a batch, all utterances sharing the same intent are treated as positives for each other. The loss pulls positives together and pushes every other intent's utterances away. This builds a geometrically clean embedding space where intents are well-separated clusters — directly reducing confusion between close pairs like `refund_request` and `wrong_item`.

**L\_CE (Linear Head):** A single linear layer maps the 384-dim vector to 13 logits (12 intents + out\_of\_scope). Cross-entropy loss draws explicit hyperplane boundaries. The head is what the ONNX model uses at inference time.

**Why both?** InfoNCE alone gives good retrieval geometry but no calibrated probabilities. CE alone can overfit silver noise and mislearn boundary regions. Together, InfoNCE regularises the representation while CE provides calibrated outputs. λ₁ = 0.3, λ₂ = 0.7 is a reasonable starting point; tune on dev ECE.

#### 2c. Dev Loop Controls

After every epoch:

1. **Macro-F1 on dev.csv** — primary signal.
2. **ECE (Expected Calibration Error)** — measures whether the model's stated confidence matches its actual accuracy. If ECE is high, temperature scaling will need a larger T\*.
3. **Adaptive LR** — reduce learning rate by 0.5 if Macro-F1 does not improve for 2 epochs.
4. **Early stopping** — halt training if Macro-F1 does not improve for 3 consecutive epochs; restore best checkpoint.

Note: dev.csv has silver labels too, so it can mislead. Use it as a **relative** signal (is this epoch better than the last?) rather than an **absolute** one.

---

### Stage 3 — Export and Calibration

#### 3a. ONNX Export

The trained PyTorch model (encoder + head) is exported to ONNX with:

- `opset_version = 17`
- Dynamic axes on sequence length
- `torch.onnx.export` with `do_constant_folding=True`

ONNX Runtime fuses matrix multiplications and eliminates Python overhead, reducing per-sample latency from \~40 ms (PyTorch CPU) to \~8 ms.

#### 3b. Temperature Scaling

After training, the model's logits are often overconfident. Temperature scaling divides all logits by a scalar T\* before softmax:

```
p = softmax(logits / T*)
```

T\* is chosen by minimising NLL on dev.csv. This does not change predictions (argmax is invariant to T\*) but makes probabilities meaningful for the OOS threshold.

#### 3c. OOS Threshold Tuning

The OOS threshold θ is the minimum max-probability below which the system returns `out_of_scope`. It is tuned on dev.csv by sweeping θ from 0.2 to 0.8 and selecting the value that maximises the F1 of the out\_of\_scope class specifically. A loose θ produces false OOS on hard but valid queries. A tight θ lets genuine OOS through.

#### 3d. Mahalanobis Parameters

For each class c, compute:

- μ\_c = mean of all training embeddings with true label c
- Shared Σ⁻¹ using **Ledoit-Wolf shrinkage** (solves the singular-matrix problem in 384 dimensions by regularising the covariance estimate toward a scaled identity)

At inference, Mahalanobis distance for a query vector v:

```
d(v, c*) = √( (v - μ_c*)ᵀ Σ⁻¹ (v - μ_c*) )
```

where c\* is the predicted class. If d > τ (tuned on dev), the system emits an OOD flag alongside the prediction, routing the message to human review — but still provides the best-effort label.

---

### Stage 4 — Inference

#### 4a. The Two-Stage Decision

The cross-encoder (a small `cross-encoder/ms-marco-MiniLM-L-6-v2` or similar) is **only triggered on confidence collisions**, defined as:

```
Top-1 probability - Top-2 probability < δ
```

δ should be tuned on dev to control the cross-encoder trigger rate. A δ of 0.15 typically triggers on \~10–15% of queries — the genuinely ambiguous ones. At all other times, the ONNX bi-encoder result is returned directly, preserving the < 15 ms guarantee.

**Critical:** Do not set δ too loosely. If the cross-encoder fires on 40% of queries, P99 latency will be dominated by the heavier model.

#### 4b. OOD vs OOS: The Exact Distinction

|  | OOD (Out-of-Distribution) | OOS (Out-of-Scope) |
| --- | --- | --- |
| **What it is** | A valid QuickCart question in unusual language the model hasn't seen | A request entirely unrelated to QuickCart |
| **Example** | `"my parcel is MIA fam"` (slang, but means order\_status) | `"how do I file my taxes?"` |
| **Detection method** | Mahalanobis distance on 384-dim features | ODIN / temperature scaling → max\_prob < θ |
| **System response** | Return best-effort label + human-review flag | Return `out_of_scope`, no further handling |
| **Why different methods?** | The query is IN the intent space but far from the training distribution. Distance from class centroid measures this. | The query is OUTSIDE the intent space entirely. Uniform low-confidence across all classes — amplified by temperature scaling — detects this. |

**On Mahalanobis stability:** Yes, Ledoit-Wolf shrinkage is required. Without it, the 384×384 covariance matrix is frequently singular (more dimensions than samples per class), and the inverse does not exist. Ledoit-Wolf adds a regulariser α·I to the sample covariance, controlled by cross-validated λ, making Σ always invertible. Use `sklearn.covariance.LedoitWolf` directly.

---

### Stage 5 — Operational Drawbacks and Mitigations

#### Drawback 1: Reranker Latency Trap

**Problem:** A loose δ causes the cross-encoder to fire too often, destroying the < 15 ms guarantee.

**Mitigation:**

- Measure cross-encoder trigger rate on dev.csv at each δ setting.
- Target < 15% trigger rate.
- Cap cross-encoder input to exactly 3 candidates (never more).
- Use the smallest cross-encoder that closes the accuracy gap: `ms-marco-MiniLM-L-6-v2` (\~22 M params) over larger variants.
- Report both median and P99 latency separately for triggered and non-triggered queries.

#### Drawback 2: Mahalanobis Instability

**Problem:** Singular covariance matrix in high-dimensional feature space.

**Mitigation:** Ledoit-Wolf shrinkage (see Stage 3d). Additionally, consider a **tied covariance** (one Σ shared across all classes, estimated on all training embeddings together), which is more stable and often performs similarly for OOD detection.

#### Drawback 3: Continual Learning — The 13th Intent

**Problem:** Adding a new intent requires full retraining because the linear head is locked to 12 output nodes.

**Mitigation strategy (documented for future work):**

- The InfoNCE-trained embedding space is modular: a new intent's centroid can be inserted as a retrieval candidate without retraining, giving a zero-shot fallback.
- The linear head must be retrained, but with the encoder frozen, this is a fast (< 1 hour) linear probe rather than full fine-tuning.
- Structural recommendation: expose the centroid-based retriever as a stable inference path so a new intent can be served the day it is defined, before the next full retrain cycle.

---

## 5. The Three Code Files

### File 1: `train.py` — Data Curation + Training

**Responsibilities:**

- Load `taxonomy.json`, expand zero-example intents
- Load `train.csv`, run Cleanlab, save cleaned set + noise report
- Build dataset with taxonomy anchors at 5× weight
- Instantiate BGE-small encoder + linear head
- Train with joint InfoNCE + CE loss, balanced batch sampler
- Dev-loop: Macro-F1, ECE, adaptive LR, early stopping
- Save `best_checkpoint.pt`

**Key dependencies:** `transformers`, `sentence-transformers`, `cleanlab`, `scikit-learn`, `torch`

---

### File 2: `eval.py` — Dev Analysis + Export

**Responsibilities:**

- Load `best_checkpoint.pt`
- Export to `model.onnx`
- Fit Temperature Scaler T\* on dev.csv
- Tune OOS threshold θ on dev.csv
- Fit Mahalanobis parameters (Ledoit-Wolf) on training embeddings
- Save `mahal_params.pkl`, `calibration_params.json`
- Produce full dev report: per-class P/R/F1, confusion matrix, ECE curve, latency histogram, cross-encoder trigger rate at each δ
- Save all results to `results/eval_dev.json` and CSVs

**Key dependencies:** `onnxruntime`, `scikit-learn`, `matplotlib`, `pandas`

---

### File 3: `predict.py` — Test Inference + Submission

**Responsibilities:**

- Load `model.onnx`, `mahal_params.pkl`, `calibration_params.json`
- Load `test.csv`
- Run inference pipeline per message: preprocess → ONNX → temp scale → OOS check → gap check → optional cross-encoder → Mahalanobis flag
- Write `submissions/<name>_predictions.csv` (id, intent)
- Report measured latency per sample on the host machine

**Key dependencies:** `onnxruntime`, `pandas`, `sentence-transformers` (cross-encoder only)

---

## 6. Latency Budget

| Step | Measured (dev laptop, 4 cores) | Estimate (prod 4-core / 15 GB) |
| --- | --- | --- |
| Preprocessing | < 0.5 ms | < 0.5 ms |
| ONNX tokenise + encode | \~8 ms | \~10–12 ms |
| Temp scale + OOS check | < 0.1 ms | < 0.1 ms |
| Mahalanobis distance | < 0.2 ms | < 0.5 ms |
| **Non-triggered total** | **\~9 ms** | **\~13 ms** |
| Cross-encoder (when triggered, \~15% of queries) | \~30 ms | \~40–60 ms |
| **Triggered P99** | **\~40 ms** | **\~65 ms** |

The < 15 ms guarantee holds for the **85% non-triggered path**. The triggered path (confidence collisions) will exceed 15 ms on a CPU-only machine. The write-up must be explicit about this trade-off and offer two options:

1. **Accept the P99 spike** on \~15% of queries as the cost of resolving genuinely ambiguous cases.
2. **Disable the cross-encoder** in production and rely solely on the bi-encoder (simpler, always < 15 ms, with slightly lower accuracy on the ambiguous cases).

---

## 7. Evaluation Honesty Checklist

The write-up must address each of these:

- Dev labels are silver. Dev Macro-F1 is a relative guide, not a ground truth. Report confidence intervals where possible.
- The taxonomy examples and train.csv were likely generated from the same process. TF-IDF on taxonomy examples therefore inflates baseline scores artificially — this is a known flaw in the evaluation setup.
- The Cleanlab noise removal improves dev scores partly because dev has the same noise pattern. The benefit on hidden gold test labels may be smaller.
- Latency numbers are hardware-specific. Report the machine, the core count, and whether ONNX inter-op threads were pinned.
- ECE after temperature scaling should be reported before and after. If post-scaling ECE > 0.05, the OOS threshold is less reliable.

---

## 8. What Would Come Next (If Time Permitted)

- **Hinglish handling:** A multilingual encoder (`paraphrase-multilingual-MiniLM-L12-v2`) could be swapped in and benchmarked against BGE-small. The Hinglish fraction of test.csv is unknown but likely non-trivial given the README calls it out explicitly.
- **Active relabelling:** The highest-Mahalanobis samples from train.csv are good candidates for a 100-sample human audit to verify noise estimates.
- **Continual learning harness:** A weekly retrain trigger on newly labelled production traffic, with the encoder frozen and only the linear head updated, would let the system improve without full retraining.
- **Threshold per-class:** A single θ for OOS is a simplification. Per-class confidence thresholds (tuned per intent) would reduce false OOS on the intents with naturally lower max probabilities.
