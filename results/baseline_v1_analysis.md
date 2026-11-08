# Baseline v1: Result Analysis

**Source:** `baseline_v1.json` (dev set, 4,000 rows, silver labels)
**Note:** The file's contents are identical to `baseline_dev.json`. The run configuration (taxonomy version, code version) is not recorded in the file. Record it before citing this as a separate experiment.

---

## 1. Overall Metrics

| Metric | Value | Count (of 4,000) | Interpretation |
|---|---|---|---|
| Top-1 accuracy | 0.5625 | 2,250 | The highest-scoring intent is correct in 56% of cases. |
| Top-3 accuracy | 0.8582 | 3,433 | The correct intent is in the top three in 86% of cases. |
| MRR | 0.7862 | n/a | On average, the correct intent is ranked close to first. |
| Macro-F1 | 0.5939 | n/a | See the caveat in section 6. |

**Top-1 to top-3 gap:** 0.2957, or 1,183 rows where the correct intent is ranked second or third. This is the main opportunity. The model usually finds the right answer, but often ranks it below a wrong one.

**Errors:** 1,750 of 4,000 rows (43.75%).

---

## 2. Per-Class Results

Sorted by F1. "Missed" is support minus correct predictions (rows the model failed to label correctly).

| Intent | Support | Precision | Recall | F1 | Correct (TP) | Missed | Tier |
|---|---|---|---|---|---|---|---|
| account_help | 441 | 0.850 | 0.644 | 0.733 | 284 | 157 | High |
| delivery_delay | 401 | 0.813 | 0.648 | 0.721 | 260 | 141 | High |
| cancel_order | 225 | 0.784 | 0.644 | 0.707 | 145 | 80 | High |
| change_address | 207 | 0.760 | 0.565 | 0.648 | 117 | 90 | Mid |
| out_of_scope | 209 | 0.730 | 0.555 | 0.630 | 116 | 93 | Mid |
| talk_to_agent | 244 | 0.627 | 0.598 | 0.612 | 146 | 98 | Mid |
| wrong_item | 525 | 0.696 | 0.524 | 0.598 | 275 | 250 | Mid |
| missing_item | 292 | 0.793 | 0.432 | 0.559 | 126 | 166 | Low |
| refund_request | 546 | 0.786 | 0.416 | 0.544 | 227 | 319 | Low |
| promo_code_issue | 320 | 0.698 | 0.441 | 0.540 | 141 | 179 | Low |
| order_status | 212 | 0.695 | 0.387 | 0.497 | 82 | 130 | Low |
| payment_failed | 378 | 0.228 | 0.876 | 0.362 | 331 | 47 | Lowest |

Tiers: High is F1 ≥ 0.70, Mid is 0.60 to 0.70, Low is 0.50 to 0.60, and Lowest is below 0.50 (`order_status` sits at the 0.497 boundary and is grouped with the Low tier above).

---

## 3. Key Findings

### 3.1 `payment_failed` is a sink for the errors

- It is predicted **1,452 times**, but only **331** are correct. That leaves **1,121 false positives**.
- Those false positives account for **64% of all errors** (1,121 of 1,750).
- Its recall is 0.876, but its precision is 0.228. A class that wins on recall while losing on precision is the signature of a bias in the scoring, not a feature the model has learned.

Where the misses go, for each class's errors that land in `payment_failed`:

| Intent | Missed rows sent to `payment_failed` | Share of its misses |
|---|---|---|
| talk_to_agent | 84 of 98 | 86% |
| delivery_delay | 101 of 141 | 72% |
| wrong_item | 174 of 250 | 70% |
| refund_request | 221 of 319 | 69% |
| promo_code_issue | 124 of 179 | 69% |
| change_address | 61 of 90 | 68% |
| order_status | 85 of 130 | 65% |
| cancel_order | 51 of 80 | 64% |
| account_help | 89 of 157 | 57% |
| out_of_scope | 51 of 93 | 55% |
| missing_item | 80 of 166 | 48% |

For most intents, around two-thirds of their misses go to `payment_failed`. That points to one shared cause, not eleven separate problems.

**Likely cause (to test, not yet confirmed):** the `payment_failed` centroid probably contains generic words such as "order", "not", "placed" and "payment", which are common across many intents. A message that shares any of those words gets a boost toward `payment_failed`. Print the top-weighted terms in each centroid to check.

### 3.2 The two "blind spots" from the old report are gone

In this run, `order_status` and `change_address` are predicted:

- `order_status`: predicted 118 times, 82 correct (precision 0.695)
- `change_address`: predicted 154 times, 117 correct (precision 0.760)

The earlier report's claim of zero predictions doesn't hold for this file. Those intents have low recall (0.387 and 0.565), not zero.

### 3.3 Precision and recall are unbalanced

| Pattern | Intents | Reading |
|---|---|---|
| High precision, low recall | missing_item (P 0.79, R 0.43), refund_request (P 0.79, R 0.42), order_status (P 0.69, R 0.39), promo_code_issue (P 0.70, R 0.44) | When the model predicts these, it is usually right. It misses most of them, mostly by sending them to `payment_failed`. |
| Balanced | account_help, delivery_delay, cancel_order | These intents have clean centroids. |
| Low precision, high recall | payment_failed (P 0.23, R 0.88) | Absorbs everything uncertain. |

Recall, not precision, is the main problem for most intents. A bias correction, not a rewrite of the taxonomy, could help more than expected for these classes. Section 5 covers how to test that.

### 3.4 `wrong_item` vs `refund_request` confusion

- 174 `wrong_item` rows go to `payment_failed`. Only 275 of 525 are correct.
- 46 `refund_request` rows go to `wrong_item`.
- Refund and wrong-item messages share vocabulary about items and bad orders. This is a real overlap, and the taxonomy examples should separate the two more sharply.

---

## 4. What Is Working

- `account_help` (F1 0.733) and `delivery_delay` (F1 0.721) have the strongest centroids. Their examples are distinct and their vocabulary is specific.
- `missing_item` precision is 0.79. When it fires, it is usually correct, so the problem is recall, not the intent's definition.
- `talk_to_agent` is less affected by the sink than most intents (about 60% of its misses go to `payment_failed`, but its recall of 0.598 is near the middle of the pack).

---

## 5. What This Implies for the Next Steps

1. **Test the bias before anything else.** Print the top-weighted terms for each centroid. If `payment_failed` is dominated by generic terms, that explains most of the sink.
2. **Try a simple per-class bias or score normalisation.** For example, subtract each centroid's mean similarity across the training set, or divide each score by its class's average. This is cheap and needs no retraining. Judge any gain by macro-F1 on dev and by how many errors it removes.
3. **Measure the ceiling of reranking.** The top-3 accuracy (0.858) is well above top-1 (0.5625). Score each row by its top-3 candidates using a stronger signal and see how much of that 1,183-row gap you can recover.
4. **Use training data.** A trained classifier (logistic regression on TF-IDF) will learn to weight distinguishing words instead of averaging them. Compare it with this baseline on the same dev set.
5. **Fix the taxonomy overlaps.** `wrong_item` vs `refund_request`, and `order_status` vs `delivery_delay` (both use "where is my order" phrasing), need explicit contrasting examples.

---

## 6. Caveats

- **Silver labels.** The dev labels contain errors that are not random. A `payment_failed` prediction that looks wrong may be the correct answer under a noisy label. Audit a sample of `payment_failed` predictions before concluding the model is at fault.
- **Macro-F1 doesn't quite match.** The mean of the twelve per-class F1 values is 0.5960, but the file reports 0.5939. The gap is small, but it means the file may not have been computed the way its name says. Recompute macro-F1 from the confusion matrix and check which averaging the code uses.
- **Dev is the measuring stick, not the target.** Any tuning against this file will overfit to dev. Keep a separate check for each change, and don't tune repeatedly against these numbers.
- **Run provenance.** The starter reported top-1 of 0.423 on the original taxonomy. This file shows 0.5625. The file doesn't say what changed (taxonomy, code, or both), so record that before comparing it with the starter.