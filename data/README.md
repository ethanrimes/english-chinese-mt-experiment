# data/

This directory holds all training and evaluation data. Everything inside `raw/`,
`interim/`, `processed/`, `splits/`, and `cache/` is **gitignored** — see the root
`.gitignore`. Only this README is tracked.

## Layout

```
data/
├── README.md          (this file — tracked)
├── raw/               (gitignored) downloaded archives, untouched
│   ├── opus/
│   ├── flores200/
│   ├── news_commentary/
│   └── ...
├── interim/           (gitignored) per-source decoded TSV/JSONL after light cleanup
├── processed/         (gitignored) post-filter, post-dedup unified parquet
│   └── all_pairs.parquet
└── splits/            (gitignored) materialized training scales + eval splits
    ├── dev.parquet
    ├── test.parquet            (FLORES-200 test, never touched in training)
    ├── scale_10k.parquet
    ├── scale_50k.parquet
    ├── scale_100k.parquet
    └── scale_500k.parquet
```

## How to populate

```
python scripts/01_download_data.py --profile full
python scripts/02_prepare_data.py
python scripts/05_create_subsets.py --sizes 10000,50000,100000,500000
```

For a quick smoke test (FLORES + News-Commentary only, ~2 minutes):

```
python scripts/01_download_data.py --profile smoke
python scripts/02_prepare_data.py --profile smoke
python scripts/05_create_subsets.py --sizes 1000,5000
```
