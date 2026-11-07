# QuickCart intent task

Welcome. This is a take-home task for the Applied ML Intern role on the Yellow.ai Research Team. Everything here is synthetic and fictional.

## The situation

QuickCart is a made-up food-delivery and grocery app. Its support chatbot reads a customer message and picks one of 12 intents by matching the message against a hard-coded **intent taxonomy** (intent names plus a few example messages each). The current matcher is a deliberately simple starter, and the system around it has problems.

Your job is to **improve the whole system** and show, with evidence, what was wrong and what helped. That includes the taxonomy, the data, the evaluation, and the model. A single accuracy number is not the goal. How deep you go and how honestly you report is.

## What you get

```
data/
  taxonomy.json   the hard-coded taxonomy: 12 intents, a few examples each
  train.csv       30,000 rows: intent, utterance   (silver labels)
  dev.csv          4,000 rows: intent, utterance   (silver labels)
  test.csv         3,000 rows: id, utterance       (labels are hidden)
baseline/
  baseline.py     taxonomy matcher: TF-IDF centroid per intent, cosine, threshold
  eval.py         evaluation on dev -> results/baseline_dev.json
results/          baseline numbers
submissions/      submission format + a format example
```

Setup (Python 3.10+):

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m baseline.eval          # dev numbers, ~10 seconds
python -m baseline.baseline      # writes submissions/baseline_predictions.csv
```

## Read this about the data

- **Labels in `train.csv` and `dev.csv` are silver, not gold.** They were produced by an automatic process and contain errors. Assume some labels are wrong, and that the errors are not random. Measuring this is part of the task.
- `test.csv` has hidden gold labels that follow the rules below. You will not see them.
- Messages are noisy on purpose: typos, abbreviations, emojis, Hinglish (mixed Hindi and English), long rambling messages, and messages that ask for more than one thing.
- The taxonomy has deliberate flaws and overlaps. Some intents are close to each other. Do not assume the starter taxonomy, the starter code, or the starter numbers are correct.

## How the hidden test labels are decided

1. The label is the action the customer wants done. If several are asked, the first one named is primary.
2. If the customer changes their mind, the last request wins ("cancel it... actually never mind, where is it?" is `order_status`).
3. Cancel plus refund: an order not yet delivered is `cancel_order`, a delivered one is `refund_request`.
4. A complaint with no request is labelled by the problem: late is `delivery_delay`, something absent is `missing_item`, something different is `wrong_item`.
5. Asking only for a human or escalation is `talk_to_agent`. If another request is also present, the other request wins.
6. Anything not about QuickCart orders, accounts or payments is `out_of_scope`. That includes small talk, gibberish, other companies' services, and services QuickCart does not offer (for example gift cards, loyalty points, tax invoices, tipping riders, restaurant feedback). A greeting or thanks alone is `out_of_scope`. A greeting plus a real request takes the request's label.
7. Filler, emotion and unrelated detail never change the label.
8. Order IDs, amounts and item names never change the label.

## What to do

There is no single right path. A strong submission usually covers:

- **Find and quantify problems.** Read real errors, group them, count them. Check the taxonomy, the labels, the evaluation code and the baseline code.
- **Improve the system.** Fix what you find. Show before and after numbers on dev (and be clear about how dev can mislead you).
- **Make choices with evidence.** If you pick an embedding model, a representation, or a threshold, show the data that backed the choice. "It is popular" is not evidence.
- **Report honestly.** Say what did not work and what you could not verify.

Finding bugs in the starter code and quantifying their effect, and proposing concrete next steps with numbers, counts as much as accuracy.

## Ground rules

- **AI tools are allowed. Open-source tools are encouraged.** Be ready to explain every line you submit and where an AI tool was wrong.
- Use open-source models that run on a laptop CPU for your final pipeline. No paid API calls in the pipeline. Fine-tuning is allowed, but treat it as a small part of the story, not the main one.
- Do not edit `test.csv`. Do not train on test. You have no test labels, so do not try to infer them.
- Expected effort is **4 to 6 hours spread over the window**. Nobody needs an all-nighter. A smaller, well-evidenced result beats a large unfocused one.

## Ideas to explore (optional, unverified)

These are starting points, not instructions. We have not tested them. Check licenses, quality and cost yourself.

- Replace TF-IDF with a sentence-embedding model. Candidates people use include MiniLM, BGE-small, E5-small and multilingual variants. Compare on your own data and report latency too.
- Measure similarity between intents (cluster or centroid distances) and relate it to your confusion matrix.
- Write short descriptions for each intent and test whether they help matching, including for intents with no examples.
- Bin the data by difficulty (length, noise, language, number of requests) and report accuracy per bin.
- Estimate label noise, for example by cross-validated disagreement or a manual audit of a sample with a confidence interval.
- Use the training data, not only the taxonomy examples. How do the two compare under the same compute?
- Report more than accuracy: per-class precision, recall and F1, macro-F1, top-k and MRR, and calibration of an "unknown" decision.
- Optional extension: handle Hinglish more deliberately. Quality beats breadth here, one thing done well is enough.

## What to submit

By **EOD Nov 9**, send the person who gave you this task:

1. A link to your GitHub repo (or a zip) with your code and a clear commit history.
2. `submissions/<your_name>_predictions.csv` in the format described in `submissions/README.md`.
3. A write-up, **maximum 2 pages**, covering: the biggest problems you found, what you changed with before and after numbers, what you tried that did not work, the limits of your evaluation, and what you would do next.

## What happens next

We score your predictions on the hidden test set (per-class precision, recall and F1, macro-F1, accuracy) and read your write-up and code. The score is one signal. The reasoning behind it matters more. Candidates who submit will get a 20 to 30 minute discussion on **Nov 10** about their work. Final results go out at the end of next week.
