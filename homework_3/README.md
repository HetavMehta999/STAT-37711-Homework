# Homework 3

Write-up: [`hw3.pdf`](hw3.pdf) (source: `writeup/hw3.tex`).

```
homework_3/
├── hw3.pdf
├── requirements.txt
├── data/          winequality-white.csv, dset0.csv, dset1.csv
├── plot_style.py  shared figure style
├── problem2/      wine_hparam_search.py, results/
├── problem3/      boosting_generalization.py, results/
└── writeup/       hw3.tex
```

Problem 1 is written only (see the PDF).

`data/winequality-white.csv` is the white-wine file from the UCI Wine Quality dataset
(<https://archive.ics.uci.edu/dataset/186/wine+quality>; `;`-separated, 4,898 rows).

## Setup
```bash
cd homework_3
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run (from `homework_3/`)
| Problem | Command | Time (laptop CPU) |
|---|---|---|
| 2(c). Grid search with train / val / test, white wine | `python problem2/wine_hparam_search.py` | ~2.5 min |
| 3. Gradient boosting with 1–1000 trees, dset0 / dset1 | `python problem3/boosting_generalization.py` | ~1 min |

Add `--quick` to either command for a fast smoke test (`--plots-only` redraws Problem 2's figures
without refitting). Each script writes metrics, LaTeX tables and figures to its `results/` folder.

## Rebuild the PDF
```bash
cd writeup && tectonic -X compile --outdir .. hw3.tex
```
(`pdflatex -output-directory=.. hw3.tex`, run twice, also works.)
