# Submission format

Put your predictions for `data/test.csv` in a CSV with exactly two columns:

```
id,intent
t00001,order_status
t00002,cancel_order
...
```

- One row per id in `data/test.csv` (3,000 rows).
- `intent` must be one of the 12 names in `data/taxonomy.json`.
- `baseline_predictions.csv` in this folder is the starter baseline's output, use it as a format example.
- Name your file `<your_name>_predictions.csv`.
