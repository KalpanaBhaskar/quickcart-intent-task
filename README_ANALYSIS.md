# Walkthrough of the QuickCart README

## What the task is

The README sets a take-home exercise for an Applied ML Intern role. You get a small, fictional support chatbot for a food-delivery app and a simple intent classifier. The job is to improve the system and show your reasoning with evidence. The README stresses that a single accuracy number is not the point. How you investigate and how honestly you report are what get judged.

**"Everything here is synthetic and fictional"** tells you that you can't look anything up about QuickCart and that the data is made up for the exercise.

**"Its support chatbot reads a customer message and picks one of 12 intents"** defines the core problem. An *intent* is the action a customer wants, such as checking an order, cancelling one, or requesting a refund. The classifier's output is a single label from a fixed list.

**"matching the message against a hard-coded intent taxonomy (intent names plus a few example messages each)"** explains the current approach. The "taxonomy" is a JSON file listing each intent with a few sample sentences. The classifier has no other knowledge of the intents.

**"The current matcher is a deliberately simple starter, and the system around it has problems."** Two things are broken: the matcher is weak, and the surroundings (taxonomy, labels, evaluation) may be flawed too. You are being asked to audit all of it.

**"Scope: ... Entity extraction, response generation, dialogue flow and UI are out of scope."** Only the classification step counts. If you find a good idea about, say, pulling out order IDs, note it as a next step but don't build it.

## Diagram 1: how the starter works

This flowchart shows the baseline pipeline, left to right.

1. **Customer message** is the input string.
2. **Clean text** normalises it, probably lowercasing and stripping punctuation. Cleaning choices can matter a lot for typos, emojis and Hinglish, so it's a place to look.
3. **Vectorise message** turns the text into numbers using TF-IDF, which weights words by how distinctive they are. The arrow shows the message going into a comparison.
4. **Taxonomy** feeds in from the side. Its intent names and examples get turned into **one vector per intent**, meaning one summary vector for each of the 12 intents.
5. **Compare message to every intent (cosine similarity)** measures the angle between the message vector and each intent vector. Scores near 1 mean the vectors point the same way, which roughly means shared words.
6. **Best score above threshold?** is the decision diamond. If the top score clears a fixed cutoff, that intent is returned. If not, the message goes to `out_of_scope`.

The diagram makes one thing clear: the only place the system can say "I don't know" is that threshold. The unknown decision is a model of its own, and it is a single fixed number here. The diagram also shows that nothing in the pipeline uses `train.csv`.

## Diagram 2: how your work is evaluated

This flowchart shows the evaluation loop, and it's more subtle.

- **taxonomy.json → Your system.** Your system can use the taxonomy.
- **train.csv → dotted arrow → Your system.** The dotted line means training data is optional. You may use it, but nothing requires it.
- **dev.csv → Your system → Dev metrics.** You get dev labels to measure yourself. The note "a guide, but noisy" is a warning. Dev labels are silver, so your dev score is an estimate and can mislead you.
- **test.csv → Your system → Predictions CSV (id, intent).** The test set has no labels. You submit predictions only.
- **Predictions CSV → We score on hidden gold labels.** The real score comes from labels you never see.
- **Your system → Write-up + prompt log → We read your reasoning.** This is a separate track for judging your process.
- **Hidden gold scoring → We read your reasoning.** Both the score and the reasoning feed into the final judgement.

The key takeaway is that you optimise against a noisy dev set, but you are graded on a clean hidden set. That gap is why the README keeps asking how dev can mislead you.

## The file layout

- **`taxonomy.json`**: the 12 intents and their example messages. This is the only thing the baseline uses.
- **`train.csv`** (30,000 rows, intent and utterance): a large labelled set, but with silver labels.
- **`dev.csv`** (4,000 rows): your measurement set, also silver.
- **`test.csv`** (3,000 rows, id and utterance): what you predict on. No labels.
- **`baseline/baseline.py`**: the starter classifier.
- **`baseline/eval.py`**: the starter evaluation. The README hints that this file needs checking, not just trusting.
- **`results/`** and **`submissions/`**: the baseline numbers and the submission format.

The setup commands just create a virtual environment, install requirements and run the baseline. `eval` is about 10 seconds, so iterating quickly is realistic.

## The baseline approach and its numbers

**The approach paragraph** describes the method in plain terms. Each intent's vector is the average of its example vectors, which is called a *centroid*. A message is scored against every centroid. If the best score is below a threshold, the answer is `out_of_scope`. The paragraph ends with the important limitation: **it does not learn from `train.csv`**. That means it uses 12 short lists of examples and nothing else.

**The results table** gives three metrics:

- **Top-1 accuracy (0.423):** the predicted intent is the correct one less than half the time.
- **Top-3 accuracy (0.719):** the correct intent is among the three highest scores about 72% of the time. A large gap from top-1 suggests the right answer is often close but ranked second or third.
- **MRR, mean reciprocal rank (0.665):** for each message, take 1 divided by the rank of the correct intent, then average. A correct first place gives 1, second gives 0.5, third gives about 0.33. It rewards getting the right answer near the top, not only at the top.

The sentence "Treat all of these as the starting point you are trying to beat, not as numbers to trust" tells you not to treat the 0.423 as a reliable baseline either. Both the model and the evaluation could be wrong.

## Reading the data honestly

- **Silver, not gold, labels.** Gold labels are checked by people. Silver labels were produced automatically, so they contain errors. The key phrase is that the errors are "not random." If mistakes cluster in certain intents or phrasings, a model can look worse than it is, or look better than it is, for the wrong reasons. Estimating label noise is an explicit part of the job.
- **Hidden test labels follow stated rules.** The gold labels come from the rules section below, so the rules are your specification.
- **Noisy messages are deliberate.** Typos, abbreviations, emojis, Hinglish, long rambling messages and multi-request messages are the realistic hard cases. A system that only works on clean English will fail the test.
- **The taxonomy is flawed on purpose.** Some intents overlap. The taxonomy is not ground truth, so you may need to rewrite it.

## The labelling rules (the hidden test standard)

These rules define the correct answer. Read them as a specification.

**Rule 1: one label, the first request decides.** Each message gets exactly one intent: the action the customer wants done. With several requests, the first one wins. The example "where is my order, and also I want a refund" is `order_status`, because it was asked first. This is the default, and the README notes that rules 2 and 5 are exceptions.

**Rule 2: the last request wins when the customer changes their mind.** "Cancel it... actually never mind, where is it?" becomes `order_status`. This is the first exception to rule 1. It means you have to track self-corrections, which a simple bag-of-words model cannot do.

**Rule 3: cancel versus refund depends on delivery status.** An undelivered order that the customer wants stopped is `cancel_order`. A delivered one is `refund_request`. The same words can map to two different labels depending on context the message may not state.

**Rule 4: complaints without a request are labelled by the problem.** Lateness is `delivery_delay`, something absent is `missing_item`, something wrong is `wrong_item`. The customer is describing a problem, not asking for an action, so the label comes from the problem type.

**Rule 5: a bare request for a human is `talk_to_agent`, but any other request beats it.** This is the second exception. "Connect me to someone" alone is `talk_to_agent`. "Connect me to someone, my refund hasn't come" is the refund intent. Escalation is a fallback, not a priority.

**Rule 6: out_of_scope is broader than it sounds.** It covers small talk, gibberish, other companies, and services QuickCart doesn't offer, such as gift cards, loyalty points, tax invoices, rider tips or restaurant reviews. A bare greeting or thanks is out of scope. A greeting plus a real request takes the request's label. Note that "services QuickCart does not offer" can sound like a QuickCart order issue, so it's a common trap.

**Rule 7: filler and emotion don't matter.** "Ugh this is so annoying, anyway..." doesn't change the label. Only the requested action counts.

**Rule 8: IDs, amounts and item names don't matter.** "Refund ₹450 for order #88213" is the same intent as "refund for order 88213." This is an instruction to be robust to entities, which are out of scope for extraction.

Rules 1, 2 and 5 are the places where a simple classifier is most likely to disagree with gold. Rules 3 and 4 also need context a bag-of-words model may not capture.

## What to do

The four bullets describe what a strong submission looks like.

- **Find and quantify problems.** Don't just list problems; count them. "Ten percent of errors are label noise" is more useful than "some labels are wrong." The README asks you to examine the taxonomy, labels, evaluation code and baseline code, so the starter's bugs count as findings.
- **Improve the system, with before and after numbers.** Each change should be measured. The README also asks you to say how dev might mislead you, which connects to the silver-label issue.
- **Make choices with evidence.** "It's popular" is not a justification. If you choose a model, show the data behind that choice.
- **Report honestly.** Failed experiments are valued. Saying what you couldn't verify builds trust.

The final sentence says that finding and quantifying bugs is as valuable as accuracy. That lowers the pressure to chase a high number.

## Compute budget

This is the hard constraint.

- **The final system must run on 4 CPU cores and 15 GB RAM,** answering one message at a time in real time. A large model that needs a GPU to respond is not allowed in production, even if it scores better.
- **GPUs are fine for experiments.** You can train or label with one. The constraint applies to what you ship.
- **State what runs where.** Separate the training environment from the production environment.
- **Report measured latency per message, the hardware, and an estimate for the 4-core box.** Measured numbers beat guesses.
- **Hosted APIs are allowed if they serve open-source models.** You must say which one, what for, and what it means in production (cost, latency, reliability). Also, do not send anything outside this repo. That's a data-handling rule, not only a cost rule.
- **Fine-tuning is allowed but should be a small part of the story.** The task rewards system thinking over training a big model.

## Ground rules

- **AI tools are encouraged, but you must understand every line you submit.** You should be able to explain it in the discussion and point out where the tool was wrong.
- **Don't edit `test.csv`, don't train on it, and don't try to infer its labels.** Trying to infer them would be cheating, even indirectly.
- **Expect 4 to 6 hours total.** A small, well-evidenced result is preferred to a big, unfocused one.
- **Ask questions.** Clarifications to Abhivanth are expected and won't count against you.

## The hints

The hints aren't instructions. Each one points at a likely weakness.

- **"Look at what the matcher actually knows about each intent."** Twelve example sentences per intent is very little. Ask whether a new human agent would have enough to tell two similar intents apart. If not, what would you add?
- **"Two intents have nothing to match against."** The baseline can't build a centroid for an intent with no examples, so those intents can never win. This is a flaw to find, and it may mean the baseline's per-class numbers are misleading for those classes.
- **"Think about which kinds of models respond in a few milliseconds."** This is a latency trade-off. Small sentence-embedding models or linear models are fast; large transformers are slow. Each has a place, perhaps a fast first pass and a slower model for hard cases.
- **"When a model and a label disagree, either could be wrong."** Don't assume the label is right. Sample disagreements and look.
- **"The taxonomy examples and the data were not created independently."** This is a hint about leakage. If the taxonomy examples came from the same generator as the data, or are copied into train or dev, the baseline's dev score could be inflated. Check overlap before trusting a score.
- **"Accuracy hides a lot."** Break results down by class, message length, language mix and number of requests. A model could be strong on average and fail on Hinglish or multi-request messages.
- **Sentence-embedding models** (MiniLM, BGE-small, E5-small, multilingual variants) are a suggested family. Compare them on this data and include latency.
- **Measure how close intents are.** If two intent centroids are very similar, that predicts confusion. Test whether the confusion matrix matches that prediction.
- **The unknown decision needs its own design.** A single threshold is a weak model of "out of scope." You could calibrate it, use a separate classifier, or test it on held-out examples.
- **Estimate label noise.** Cross-validated disagreement or a manual audit with a confidence interval are both valid. A manual audit of a random sample, with a proper interval, is strong evidence.
- **Use training data, and compare it with taxonomy examples under the same compute.** This makes the comparison fair.
- **Handling Hinglish is optional.** One thing done well is enough.

## What to submit

1. **A repo or zip with a clear commit history.** The history shows how your thinking developed, so don't squash it into one commit.
2. **`submissions/<your_name>_predictions.csv`**, in the format described in `submissions/README.md`. Check that format carefully; a malformed file may not be scored.
3. **A write-up of at most 2 pages.** It must cover the biggest problems, what you changed with before and after numbers, what failed, the limits of your evaluation, latency and compute notes, and next steps. Two pages is tight, so prioritise.
4. **Results as CSVs or simple tables.** Use one row per experiment or per class, clear column names, and before and after comparisons. The README says clarity of communication is judged, so make the tables easy to read.
5. **A prompt log (`prompt_log.md` or an exported chat).** List every prompt, in order, with a one-line note on what you did with the answer. Mark wrong or rejected answers. The README says the log is for understanding how you work, not for catching mistakes, so keep it honest and untidy.

## What happens next

Your predictions are scored on the hidden test set using per-class precision, recall and F1, macro-F1 and accuracy. Macro-F1 gives each class equal weight, so weak performance on rare intents will hurt it. Then your write-up, prompt log and code are read. Shortlisted candidates have a 20 to 30 minute discussion on November 10, so you should be ready to defend every decision. Final results come at the end of the following week.

## The overall picture

The README is testing four things, in order of importance to them:

1. **Can you find real problems** in the taxonomy, labels, evaluation and code, rather than only tuning a model?
2. **Can you measure them** and show evidence for each choice?
3. **Can you ship something realistic** under a strict CPU budget and real-time latency?
4. **Are you honest** about what worked, what didn't, and what you couldn't verify?

If you want, I can help you turn this into a concrete plan, such as which analyses to run first on the baseline and dev set.