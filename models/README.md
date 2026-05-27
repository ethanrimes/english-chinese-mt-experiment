# models/

Storage for model artifacts. Everything except this README is **gitignored**.

```
models/
├── cached/           base SmolLM2-360M (downloaded from HF, not redistributed)
├── extended/         base + extended tokenizer + resized embeddings (the starting checkpoint)
└── runs/             one subdir per training run
    └── scale-<N>-<timestamp>/
        ├── train.log
        ├── metrics.jsonl
        ├── tb/                       TensorBoard
        ├── checkpoint-<step>/        intermediate checkpoints
        ├── best/                     best-dev-BLEU checkpoint (symlink/copy)
        └── final/                    end-of-training checkpoint
```

Checkpoints are large (~1.5 GB each for the extended model) — they live here locally and
are mirrored to Azure Blob via the AzureML output mount during cloud runs.
