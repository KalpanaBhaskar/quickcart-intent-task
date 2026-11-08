confusion matrix from baseline_dev.json

| Actual \ Predicted | 1. status | 2. delay | 3. cancel | 4. refund | 5. missing | 6. wrong | 7. pay_fail | 8. promo | 9. address | 10. account | 11. agent | 12. OOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. order_status** | **0** | 3 | 14 | 1 | 2 | 3 | 178 | 0 | 0 | 1 | 9 | 1 |
| **2. delivery_delay** | 0 | **214** | 1 | 19 | 0 | 32 | 123 | 1 | 0 | 4 | 5 | 2 |
| **3. cancel_order** | 0 | 3 | **118** | 0 | 1 | 3 | 80 | 1 | 0 | 3 | 1 | 15 |
| **4. refund_request** | 0 | 7 | 4 | **144** | 3 | 51 | 292 | 3 | 0 | 2 | 11 | 29 |
| **5. missing_item** | 0 | 4 | 0 | 2 | **81** | 10 | 148 | 2 | 0 | 5 | 6 | 34 |
| **6. wrong_item** | 0 | 1 | 4 | 18 | 2 | **308** | 178 | 1 | 0 | 4 | 6 | 3 |
| **7. payment_failed** | 0 | 7 | 4 | 2 | 2 | 4 | **345** | 1 | 0 | 5 | 4 | 4 |
| **8. promo_code_issue** | 0 | 3 | 0 | 4 | 1 | 5 | 206 | **86** | 0 | 8 | 6 | 1 |
| **9. change_address** | 0 | 1 | 5 | 1 | 1 | 3 | 188 | 4 | **0** | 0 | 4 | 0 |
| **10. account_help** | 0 | 9 | 3 | 4 | 4 | 5 | 181 | 13 | 0 | **217** | 4 | 1 |
| **11. talk_to_agent** | 0 | 3 | 1 | 2 | 0 | 1 | 142 | 1 | 0 | 2 | **92** | 0 |
| **12. out_of_scope** | 0 | 5 | 2 | 0 | 2 | 4 | 100 | 0 | 0 | 2 | 6 | **88** |


# Intent Classifier Performance Analysis Report

This report provides a comprehensive analysis of the intent classifier based on the evaluation metrics, per-class performance data, and confusion matrix provided. 

---

## 1. Overall System Metrics

*   **Top-1 Accuracy: 42.33%** 
    *   **Interpretation:** The model's absolute highest-scoring prediction is correct less than half the time. Out of 4,000 queries, it hits the exact correct intent on the first attempt only **1,693 times**.
*   **Top-3 Accuracy: 71.92%**
    *   **Interpretation:** When looking at the model's top 3 choices, the correct intent is present **2,876 times**. This significantly higher score indicates the correct intent is almost always nearby, even when the top pick is wrong.
*   **Mean Reciprocal Rank (MRR): 0.6652**
    *   **Interpretation:** On average, the correct intent sits between the 1st and 2nd rank (closer to position 1.5). The model is rarely completely off-base; it is simply struggling to differentiate its top two choices.
*   **Macro F1-Score: 0.4460**
    *   **Interpretation:** This unweighted average of F1-scores across all classes is quite low, signaling that performance is highly unstable and unequal across different intents.

---

## 2. Confusion Matrix Key Discoveries

The 12x12 grid reveals structural imbalances and a critical model bottleneck:

### 🚨 The "Black Hole" Effect: `payment_failed` (Column 7)
The model has developed a severe, systemic bias toward predicting `payment_failed`. It acts as a default bucket whenever the classifier encounters uncertainty.
*   **Massive Misclassifications:** The model incorrectly dumps **292** `refund_request` queries, **206** `promo_code_issue` queries, **188** `change_address` queries, and **178** `order_status` queries straight into `payment_failed`.
*   **Statistical Impact:** Because it over-predicts this class aggressively, `payment_failed` achieves a massive **Recall of 91.27%** but a terrible **Precision of 15.96%**. 

### 🚫 Completely Blind Spots (Columns 1 & 9)
The model completely fails to predict certain intents.
*   **Zero Predictions:** Both `order_status` (Column 1) and `change_address` (Column 9) contain **absolute zeros** across all rows. 
*   **Statistical Impact:** The model predicted these two intents exactly **0 times** out of 4,000 opportunities, resulting in an unresolvable `0.0` for Precision, Recall, and F1-score.

### 📉 Silver Label Vulnerability (Data Noise)
*   The data note indicates `"dev labels are silver (noisy)"`. 
*   Given how uniform the misclassifications into `payment_failed` are, it is highly likely that the automated script used to generate this validation data incorrectly cross-labeled various intents as `payment_failed`, penalizing the model even when it might be correct.

---

## 3. Per-Class Tier Breakdown

Based on F1-scores, the 12 intents fall cleanly into three performance tiers:

### 🟢 Tier 1: Strong Performers (F1 > 0.60)
These intents have relatively strong boundaries. The model handles these consistently well.
*   **`delivery_delay` (F1: 0.6475):** Good balance, capturing over half the actual cases with high precision (82.31%).
*   **`wrong_item` (F1: 0.6457):** The highest raw count of true positives (**308** correct hits).
*   **`account_help` (F1: 0.6254):** Solid precision (85.77%), meaning when it predicts this, it is usually right.
*   **`cancel_order` (F1: 0.6194):** Standard acceptable performance.

### 🟡 Tier 2: Struggling / Low Recall (F1: 0.25 - 0.50)
These intents suffer heavily because the model fails to catch them, leaking the majority of their volume to the `payment_failed` bottleneck.
*   **`talk_to_agent` (F1: 0.4623)**
*   **`out_of_scope` (F1: 0.4548)**
*   **`missing_item` (F1: 0.4143):** High precision (81.82%) but abysmal recall (27.74%). The model knows what it looks like but misses most of them.
*   **`promo_code_issue` (F1: 0.3972):** Misses **206** out of 320 instances to `payment_failed`.
*   **`refund_request` (F1: 0.3876):** Severe leakage. Misses **292** out of 546 instances to `payment_failed`.
*   **`payment_failed` (F1: 0.2718):** Low F1 due to the flood of false positives lowering its precision.

### 🔴 Tier 3: Catastrophic Failure (F1: 0.00)
*   **`order_status` (F1: 0.0000)**
*   **`change_address` (F1: 0.0000)**
*   *Diagnosis:* The model has effectively deleted these intents from its vocabulary.

---

## 4. Next Actions & Recommendations

1.  **Audit the Silver Data:** Before retraining the model, manually check a sample of validation data labeled `payment_failed`. Verify if they are actual payment errors or if they are mislabeled `refund_request` / `order_status` queries.
2.  **Apply Class Weights / Penalties:** Introduce class balancing weights during training to heavily penalize the model for guessing `payment_failed` incorrectly, forcing it to distribute predictions to underrepresented classes.

