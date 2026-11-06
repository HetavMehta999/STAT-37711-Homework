# Homework 2

Write-up: [`hw2.pdf`](hw2.pdf) (source: `writeup/hw2.tex`).

```
homework_2/
├── hw2.pdf
├── requirements.txt
├── data/          taxi_trips.parquet, retail.csv, train.csv, val.csv, test.csv
├── problem1/      taxi_mlp.py, results/
├── problem2/      retail_encoding.py, results/
├── problem3/      rks_vs_mlp.py, models.py, dataloader.py, plotting.py,
│                  rks_test_predictions.csv, mlp_test_predictions.csv, results/
└── writeup/       hw2.tex
```

## Setup
```bash
cd homework_2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run (from `homework_2/`)
| Problem | Command | Time (laptop CPU) |
|---|---|---|
| 1. Heteroscedastic MLP, taxi trips | `python problem1/taxi_mlp.py` | ~1.5 min |
| 2. Target encoding vs. embeddings, retail | `python problem2/retail_encoding.py` | ~1.5 min |
| 3. Random Kitchen Sinks vs. MLP | `python problem3/rks_vs_mlp.py` | ~30 s |

Add `--quick` to any command for a fast smoke test. Each script writes metrics, LaTeX tables and
figures to its `results/` folder. Problem 3 also writes `problem3/rks_test_predictions.csv` and
`problem3/mlp_test_predictions.csv` (one prediction per row, same order as `data/test.csv`, no header).

## Rebuild the PDF
```bash
cd writeup && pdflatex -output-directory=.. hw2.tex && pdflatex -output-directory=.. hw2.tex
```
