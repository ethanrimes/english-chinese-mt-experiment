# english-chinese-mt-experiment

Teaching a small **English-only** language model (SmolLM2-360M) to translate English ↔ Chinese
via supervised fine-tuning on parallel data. Designed as a **data-scaling study**: we train
**six separate models**, one per data scale (10K / 50K / 100K / 500K / 1M / 5M parallel
pairs), with identical hyperparameters (epoch count tuned per scale to keep total examples
seen comparable). Within each model's training run we save checkpoint snapshots every
`save_steps` so we can trace quality-over-training for every scale. The six resulting
curves give us a quality-vs-data picture when all Chinese linguistic knowledge must be
acquired from scratch.

## Why this is interesting

- The base model **has zero Chinese in its pretraining**. Every character, every grammatical
  pattern, every vocabulary item must come from the fine-tuning data.
- We **extend the tokenizer** with ~16K Chinese BPE tokens (initialized fresh) while
  preserving the original English vocabulary verbatim. This isolates "learning a language
  from zero" cleanly.
- The data-scaling sweep gives us a quality-vs-data curve, with the same compute spent at
  each scale (epochs, not steps, held constant on the largest set).

## Repo layout

```
configs/        YAML configs: model, data sources, training, scales
scripts/        CLI entry points (download, prep, train, eval, submit)
src/ecmt/       Library code (data, model, training, utils)
azure/          Azure ML environment/compute/job specs
tests/          Pytest unit tests
data/           (gitignored) raw / interim / processed / splits
models/         (gitignored) base + extended checkpoints, run outputs
reports/        Generated eval reports (gitignored: figures, runs)
```

## Quickstart (local, smoke-test scale)

```powershell
# 1. Environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 2. Download a small sample (FLORES + News-Commentary only)
python scripts/01_download_data.py --profile smoke

# 3. Prepare + filter + dedupe
python scripts/02_prepare_data.py --profile smoke

# 4. Build the data-scale subsets
python scripts/05_create_subsets.py --sizes 1000,5000

# 5. Train the tokenizer extension on Chinese side
python scripts/03_train_tokenizer_extension.py --vocab-size 8000

# 6. Extend the base model (resize embeddings)
python scripts/04_extend_model.py

# 7. Train at smoke scale
python scripts/06_train.py --config configs/training.yaml --scale 1000 --epochs 1

# 8. Eval
python scripts/07_evaluate.py --run-dir models/runs/scale-1000
```

## Full study on Azure ML

```powershell
# Provision once
az ml workspace create -f azure/workspace.yaml
az ml compute create -f azure/compute.yaml -w <workspace>

# Build environment
az ml environment create -f azure/environment.yaml -w <workspace>

# Submit the full data-scaling sweep (4 runs in parallel)
python scripts/submit_aml_job.py --sweep configs/sweep_data_scale.yaml
```

See [PLAN.md](PLAN.md) for the full experimental plan, framework choices, and Azure
resource sizing.

## Observability

- **Weights & Biases** + **MLflow** (Azure ML native) — both enabled by default.
- Per-run TensorBoard logs in `models/runs/<run-id>/tb/`.
- Checkpoints every `save_steps` (default 500) — last K kept + every eval milestone retained
  permanently. Catastrophic-forgetting probe (English-only perplexity) logged each eval to
  flag when English ability degrades faster than Chinese improves.
