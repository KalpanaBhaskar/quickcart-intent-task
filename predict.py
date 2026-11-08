"""
predict.py — QuickCart Intent Pipeline  |  Phase 4 (Inference + Submission)
=============================================================================

What this file does
-------------------
1. Load model.onnx (ONNX Runtime, 4-core CPU).
2. Load calibration_params.json  (T*, θ, δ, τ, label map).
3. Load mahal_params.pkl         (centroids, precision matrix, τ).
4. Load test.csv (id, utterance — no labels).
5. For each message, run the full 4-stage inference pipeline:
     a. Preprocess
     b. ONNX encode → logits + embedding
     c. Temperature scale → calibrated probabilities
     d. OOS check: max_prob < θ → out_of_scope
     e. Confidence gap check: gap < δ → trigger cross-encoder on top-K
     f. Mahalanobis distance: dist > τ → emit OOD flag (still returns label)
6. Write submissions/<name>_predictions.csv with columns [id, intent].
7. Report per-sample latency statistics.

Production notes
----------------
- The ONNX path (steps a–d, f) runs in ~10–13 ms on 4 CPU cores (P50).
- The cross-encoder (step e) adds ~30–60 ms and fires on ~15% of queries.
- The P99 latency on the non-triggered path is <15 ms (within the ceiling).
- The triggered path will exceed 15 ms on a 4-core box; see config.py and the
  write-up for the trade-off discussion.

Run
---
  python -m predict

Dependencies: onnxruntime, sentence-transformers, pandas, numpy, tqdm
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
from tqdm import tqdm
from transformers import AutoTokenizer

from config import (
    TEST_CSV, ONNX_MODEL_PATH, MAHAL_PARAMS_PATH, CALIBRATION_PARAMS_PATH,
    SUBMISSION_CSV, LOGS_DIR, RESULTS_DIR,
    INTENT_LABELS, NUM_CLASSES, LABEL2IDX, IDX2LABEL, OOS_IDX,
    ENCODER_MODEL_NAME, MAX_SEQ_LEN,
    CROSS_ENCODER_MODEL, CROSS_ENCODER_TOP_K,
)
from utils import get_logger, preprocess, has_confidence_collision, mahalanobis_distance

logger = get_logger(__name__, LOGS_DIR / "predict.log")


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------

def load_artefacts() -> tuple[ort.InferenceSession, object, dict, dict]:
    """
    Load all inference artefacts: ONNX session, tokenizer, calibration
    parameters, and Mahalanobis parameters.

    Returns:
        (ort_session, tokenizer, calibration, mahal_params)
    """
    # Verify artefacts exist
    for p in (ONNX_MODEL_PATH, CALIBRATION_PARAMS_PATH, MAHAL_PARAMS_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"Required artefact not found: {p}. "
                f"Run train.py then eval.py first."
            )

    # --- ONNX Runtime session (same config as eval.py) ---
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3  # suppress verbose logs

    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_NAME)

    # --- Calibration parameters ---
    with open(CALIBRATION_PARAMS_PATH) as f:
        calibration = json.load(f)

    logger.info(
        f"Loaded calibration: T*={calibration['T_star']:.4f}, "
        f"θ={calibration['theta']}, "
        f"δ={calibration['delta']}, "
        f"τ={calibration['tau']:.4f}"
    )

    # --- Mahalanobis parameters ---
    with open(MAHAL_PARAMS_PATH, "rb") as f:
        mahal_params = pickle.load(f)

    logger.info(
        f"Loaded Mahalanobis params: "
        f"{len(mahal_params['centroids'])} centroids, "
        f"precision matrix shape={mahal_params['precision'].shape}"
    )

    return session, tokenizer, calibration, mahal_params


# ---------------------------------------------------------------------------
# Single-sample inference: the full 4-stage pipeline
# ---------------------------------------------------------------------------

class IntentPredictor:
    """
    Encapsulates the complete inference pipeline for a single customer message.

    Instantiate once at startup; call predict() for each message.

    Args:
        session:     ONNX Runtime InferenceSession.
        tokenizer:   HuggingFace tokenizer.
        calibration: Dict from calibration_params.json.
        mahal_params: Dict from mahal_params.pkl.
        use_cross_encoder: If True, trigger the cross-encoder on confidence
                     collisions.  Set to False to stay always under 15 ms.
    """

    def __init__(
        self,
        session: ort.InferenceSession,
        tokenizer,
        calibration: dict,
        mahal_params: dict,
        use_cross_encoder: bool = True,
    ) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self.T_star = float(calibration["T_star"])
        self.theta = float(calibration["theta"])   # OOS threshold
        self.delta = float(calibration["delta"])   # confidence gap threshold
        self.tau = float(calibration["tau"])        # Mahalanobis OOD threshold
        self.oos_idx = OOS_IDX
        self.centroids = mahal_params["centroids"]
        self.precision = mahal_params["precision"]
        self.mahal_tau = mahal_params.get("tau", self.tau)

        self.use_cross_encoder = use_cross_encoder
        self._cross_encoder = None   # lazy load

        # Build an intent description map for cross-encoder input.
        # The cross-encoder scores (message, intent_description) pairs; a richer
        # description gives better discrimination than the bare intent name.
        self.intent_descriptions = self._build_intent_descriptions()

    # ------------------------------------------------------------------
    # Intent descriptions for cross-encoder scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _build_intent_descriptions() -> dict[int, str]:
        """
        Map each class index to a natural-language description used as the
        second input to the cross-encoder (message, description).

        More informative descriptions → better cross-encoder discrimination
        on confusable pairs (e.g. refund_request vs wrong_item vs missing_item).
        """
        return {
            LABEL2IDX["order_status"]:     "Customer asking where their order is, tracking status, or delivery location",
            LABEL2IDX["delivery_delay"]:   "Customer complaining that their delivery is late or taking too long",
            LABEL2IDX["cancel_order"]:     "Customer wants to cancel an order that has not yet been delivered",
            LABEL2IDX["refund_request"]:   "Customer requesting a refund or money back for a delivered or charged order",
            LABEL2IDX["missing_item"]:     "Customer's order arrived but one or more items are missing from the bag",
            LABEL2IDX["wrong_item"]:       "Customer received the wrong item, wrong food, or someone else's order",
            LABEL2IDX["payment_failed"]:   "Customer has a payment that failed, was declined, or was charged incorrectly",
            LABEL2IDX["promo_code_issue"]: "Customer has a problem applying a promo code, coupon, or discount",
            LABEL2IDX["change_address"]:   "Customer wants to change the delivery address for an order",
            LABEL2IDX["account_help"]:     "Customer has a problem with their account: login, OTP, password, profile",
            LABEL2IDX["talk_to_agent"]:    "Customer asking to speak to a human agent or customer service representative",
            OOS_IDX:                       "Message is not related to QuickCart orders, accounts, or payments",
        }

    # ------------------------------------------------------------------
    # Cross-encoder lazy load
    # ------------------------------------------------------------------

    def _get_cross_encoder(self):
        """
        Lazily load the cross-encoder to avoid import overhead when it won't
        be triggered.
        """
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
            self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        return self._cross_encoder

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(
        self,
        raw_text: str,
    ) -> dict:
        """
        Run the full inference pipeline on a single customer message.

        Pipeline stages:
          1. Preprocess (Unicode norm, lowercase, whitespace)
          2. Tokenise + ONNX encode → raw logits + embedding
          3. Temperature scale → calibrated probabilities
          4. OOS check: max_prob < θ → return out_of_scope
          5. Confidence gap check: gap < δ → cross-encoder rerank top-K
          6. Mahalanobis distance check → OOD flag (does not change label)

        Args:
            raw_text: Raw customer message string.

        Returns:
            dict with:
              intent:       Predicted intent label string.
              confidence:   Max calibrated probability (float).
              oos_flagged:  True if max_prob < θ.
              ood_flagged:  True if Mahalanobis distance > τ.
              mahal_dist:   Mahalanobis distance to predicted class centroid.
              cross_encoder_triggered: True if gap check fired.
              latency_ms:   End-to-end latency in milliseconds.
        """
        t_start = time.perf_counter()

        # ---- Stage 1: Preprocess ----------------------------------------
        text = preprocess(raw_text)

        # ---- Stage 2: ONNX encode ---------------------------------------
        enc = self.tokenizer(
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
        logits_raw, embedding = self.session.run(["logits", "embeddings"], feed)
        logits_raw = logits_raw[0]    # [NUM_CLASSES]
        embedding = embedding[0]      # [EMBEDDING_DIM]

        # ---- Stage 3: Temperature scaling --------------------------------
        scaled = logits_raw / self.T_star
        shifted = scaled - scaled.max()
        exp_s = np.exp(shifted)
        probs = exp_s / exp_s.sum()   # [NUM_CLASSES] calibrated probabilities

        max_prob = float(probs.max())
        pred_idx = int(probs.argmax())

        # ---- Stage 4: OOS check -----------------------------------------
        oos_flagged = max_prob < self.theta
        if oos_flagged:
            intent = "out_of_scope"
            cross_encoder_triggered = False
        else:
            # ---- Stage 5: Confidence gap check + cross-encoder ----------
            cross_encoder_triggered = (
                self.use_cross_encoder
                and has_confidence_collision(probs, self.delta)
            )

            if cross_encoder_triggered:
                # Get top-K candidate class indices
                top_k_indices = np.argsort(probs)[::-1][:CROSS_ENCODER_TOP_K].tolist()
                intent = self._rerank_with_cross_encoder(text, top_k_indices)
                pred_idx = LABEL2IDX.get(intent, pred_idx)
            else:
                intent = IDX2LABEL[pred_idx]

        # ---- Stage 6: Mahalanobis OOD flag ------------------------------
        centroid = self.centroids.get(
            pred_idx,
            np.zeros_like(embedding, dtype=np.float32),
        )
        mahal_dist = mahalanobis_distance(
            embedding.astype(np.float64),
            centroid.astype(np.float64),
            self.precision.astype(np.float64),
        )
        ood_flagged = mahal_dist > self.mahal_tau

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "intent": intent,
            "confidence": max_prob,
            "oos_flagged": oos_flagged,
            "ood_flagged": ood_flagged,
            "mahal_dist": round(mahal_dist, 4),
            "cross_encoder_triggered": cross_encoder_triggered,
            "latency_ms": round(latency_ms, 3),
        }

    def _rerank_with_cross_encoder(
        self,
        text: str,
        candidate_indices: list[int],
    ) -> str:
        """
        Score (message, intent_description) pairs with the cross-encoder and
        return the top-scoring intent label.

        The cross-encoder is a small BERT-style model that jointly attends to
        the message and the intent description — it has much higher accuracy on
        ambiguous cases than the bi-encoder, at the cost of ~30 ms latency.

        Args:
            text:              Preprocessed customer message.
            candidate_indices: Top-K class indices from bi-encoder softmax.

        Returns:
            Best intent label string.
        """
        ce = self._get_cross_encoder()

        # Build (query, document) pairs for the cross-encoder
        pairs = [
            (text, self.intent_descriptions.get(idx, IDX2LABEL.get(idx, "unknown")))
            for idx in candidate_indices
        ]
        scores = ce.predict(pairs)  # float array of length K

        best_local_idx = int(np.argmax(scores))
        best_global_idx = candidate_indices[best_local_idx]
        return IDX2LABEL.get(best_global_idx, "out_of_scope")


# ---------------------------------------------------------------------------
# Batch inference on test.csv
# ---------------------------------------------------------------------------

def run_batch_inference(
    predictor: IntentPredictor,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the predictor on every row of test.csv and collect latency statistics.

    Args:
        predictor: Configured IntentPredictor instance.
        test_df:   test.csv DataFrame with columns [id, utterance].

    Returns:
        (predictions_df, latency_stats)
          predictions_df: DataFrame with columns [id, intent].
          latency_stats:  dict of P50/P95/P99 latency in ms.
    """
    ids = test_df["id"].tolist()
    utterances = test_df["utterance"].fillna("").tolist()

    intents = []
    latencies = []
    oos_count = 0
    ood_count = 0
    ce_triggered_count = 0

    for uid, utt in tqdm(
        zip(ids, utterances), total=len(ids), desc="Predicting test set"
    ):
        result = predictor.predict(utt)
        intents.append(result["intent"])
        latencies.append(result["latency_ms"])

        if result["oos_flagged"]:
            oos_count += 1
        if result["ood_flagged"]:
            ood_count += 1
        if result["cross_encoder_triggered"]:
            ce_triggered_count += 1

    n = len(ids)
    logger.info(
        f"Inference complete: {n} samples. "
        f"OOS: {oos_count} ({100*oos_count/n:.1f}%), "
        f"OOD flags: {ood_count} ({100*ood_count/n:.1f}%), "
        f"CE triggered: {ce_triggered_count} ({100*ce_triggered_count/n:.1f}%)"
    )

    latencies_arr = np.array(latencies)
    latency_stats = {
        "median_ms": float(np.median(latencies_arr)),
        "p50_ms":    float(np.percentile(latencies_arr, 50)),
        "p95_ms":    float(np.percentile(latencies_arr, 95)),
        "p99_ms":    float(np.percentile(latencies_arr, 99)),
        "mean_ms":   float(latencies_arr.mean()),
        "n_samples": n,
        "oos_count": oos_count,
        "ood_count": ood_count,
        "ce_triggered_count": ce_triggered_count,
    }
    logger.info(
        f"Latency: median={latency_stats['median_ms']:.1f} ms, "
        f"P95={latency_stats['p95_ms']:.1f} ms, "
        f"P99={latency_stats['p99_ms']:.1f} ms"
    )

    predictions_df = pd.DataFrame({"id": ids, "intent": intents})
    return predictions_df, latency_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Entry point: load artefacts, run test inference, write submission CSV.
    """
    logger.info("=" * 60)
    logger.info("PHASE 4 — INFERENCE AND SUBMISSION")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Load artefacts
    # -----------------------------------------------------------------------
    session, tokenizer, calibration, mahal_params = load_artefacts()

    # -----------------------------------------------------------------------
    # Build predictor
    # -----------------------------------------------------------------------
    predictor = IntentPredictor(
        session=session,
        tokenizer=tokenizer,
        calibration=calibration,
        mahal_params=mahal_params,
        use_cross_encoder=True,  # Set False for always-<15ms mode
    )

    # -----------------------------------------------------------------------
    # Load test.csv
    # -----------------------------------------------------------------------
    logger.info(f"Loading test data from {TEST_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"test.csv not found at {TEST_CSV}")

    test_df = pd.read_csv(TEST_CSV)
    if "id" not in test_df.columns or "utterance" not in test_df.columns:
        raise ValueError(
            f"test.csv must have 'id' and 'utterance' columns. "
            f"Found: {list(test_df.columns)}"
        )
    logger.info(f"Test rows: {len(test_df)}")

    # -----------------------------------------------------------------------
    # Warm-up: 10 silent forward passes to initialise ONNX Runtime JIT
    # -----------------------------------------------------------------------
    logger.info("Warming up ONNX Runtime (10 passes)…")
    warmup_texts = test_df["utterance"].fillna("").head(10).tolist()
    for t in warmup_texts:
        predictor.predict(t)

    # -----------------------------------------------------------------------
    # Batch inference
    # -----------------------------------------------------------------------
    predictions_df, latency_stats = run_batch_inference(predictor, test_df)

    # -----------------------------------------------------------------------
    # Save submission CSV
    # -----------------------------------------------------------------------
    SUBMISSION_CSV.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(SUBMISSION_CSV, index=False)
    logger.info(f"Saved submission → {SUBMISSION_CSV}")
    logger.info(f"Submission shape: {predictions_df.shape}")
    logger.info(f"Intent distribution:\n{predictions_df['intent'].value_counts().to_string()}")

    # -----------------------------------------------------------------------
    # Save latency report
    # -----------------------------------------------------------------------
    latency_report_path = RESULTS_DIR / "test_latency_report.json"
    latency_stats["hardware_note"] = (
        "ONNX Runtime, CPUExecutionProvider, 4 intra-op threads. "
        "P99 includes cross-encoder when triggered (~15% of queries)."
    )
    with open(latency_report_path, "w") as f:
        json.dump(latency_stats, f, indent=2)
    logger.info(f"Saved latency report → {latency_report_path}")

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("INFERENCE SUMMARY")
    logger.info(f"  Test samples  : {latency_stats['n_samples']}")
    logger.info(f"  OOS predicted : {latency_stats['oos_count']}")
    logger.info(f"  OOD flagged   : {latency_stats['ood_count']}  (review flag only; label still returned)")
    logger.info(f"  CE triggered  : {latency_stats['ce_triggered_count']}")
    logger.info(f"  Median latency: {latency_stats['median_ms']:.1f} ms")
    logger.info(f"  P99 latency   : {latency_stats['p99_ms']:.1f} ms")
    logger.info(f"  Submission    : {SUBMISSION_CSV}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
