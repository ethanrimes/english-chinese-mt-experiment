# Experimental Plan

## 1. Research question

> **How does English↔Chinese translation quality scale with parallel-data volume when the
> base LM has zero Chinese in its pretraining?**

Operationalized as: train **four separate models** of SmolLM2-360M, one per data scale
— 10K / 50K / 100K / 500K parallel pairs (nested subsets, same hyperparameters per run).
Within each training run we save a **time-series of checkpoint snapshots** (every
`save_steps`), evaluate each snapshot on FLORES-200 dev/test in both directions
(BLEU / chrF++ / COMET), and also probe English-only perplexity to detect catastrophic
forgetting. The deliverable is four learning curves — one per data scale — that together
form the quality-vs-data picture.

## 2. Base model — SmolLM2-360M

| Property | Value |
|---|---|
| HuggingFace ID | `HuggingFaceTB/SmolLM2-360M` |
| Parameters | 362M |
| Hidden | 960 |
| Layers | 32 |
| Heads | 15 (5 KV) |
| Vocab | 49,152 (GPT-2-style byte-level BPE) |
| Context | 8,192 |
| Training corpus | FineWeb-Edu (English-filtered) + Cosmopedia v2 + StackEdu + py-edu |
| Chinese content | None documented; FineWeb-Edu is language-filtered to English |

We treat the 49,152-token vocabulary as the "frozen English alphabet" and append a fresh
~16K-token Chinese sub-vocabulary on top.

## 3. Tokenizer extension

1. Concatenate the **Chinese side** of all parallel data (after filtering).
2. Train a SentencePiece BPE model on that text only, vocab size 16,000, character coverage
   0.9995, byte-fallback enabled (so any rare CJK still encodes).
3. Filter the resulting pieces to **CJK-only tokens** (drop any piece that overlaps with the
   base SmolLM2 vocabulary by more than `whitespace`).
4. Append surviving pieces as new tokens to the SmolLM2 tokenizer (`add_tokens(...)`).
5. Also add special tokens: `<|en2zh|>`, `<|zh2en|>`, `<|src|>`, `<|tgt|>`.
6. Resize the model's input embedding matrix and LM head; initialize new rows
   `N(0, 0.02)`. Original embedding rows untouched.

Why this matters: without extension, each Chinese character costs 3 bytes ≈ 3-6 tokens of
byte-level BPE — training is 3-6× more expensive per Chinese character and the model
spends most of its capacity learning to reassemble bytes into characters.

## 4. Data — sources and scale

### 4.1 Sources (in priority order)

| Source | Domain | Approx pairs | License | Use |
|---|---|---|---|---|
| **FLORES-200** | Wikipedia-like, professionally translated | 2K dev + 1K test | CC-BY-SA | **Eval only** — never in train |
| **News-Commentary v18** | Newswire, formal register | ~313K zh-en | CC-BY-NC | Train + dev |
| **TED2020 (OPUS)** | Talks, spoken-formal register | ~400K zh-en | CC-BY-NC-ND | Train |
| **WikiMatrix v1 (OPUS)** | Wikipedia-mined | ~2M (raw) | CC-BY-SA | Train, heavy filtering |
| **OpenSubtitles 2018 (OPUS)** | Movie subtitles, colloquial | ~10M (raw) | OPUS terms | Train, sampled |
| **UN Parallel Corpus v1.0** | UN proceedings, very formal | ~15M | UN terms | Train, sampled |

### 4.2 Filtering pipeline

Applied in order to every source:

1. **Length filter** — 3 ≤ tokens(both sides) ≤ 200; src/tgt ratio ∈ [0.5, 2.0].
2. **Language ID** — `fasttext` lid.176 must say `en` for the English side and `zh` for the
   Chinese side. Confidence ≥ 0.7.
3. **Script ratio** — English side ≥ 80% ASCII letters; Chinese side ≥ 30% Han characters.
4. **Dedup** — exact dedup on (src, tgt); also dedup against FLORES-200 dev+test by
   normalized-text hash to prevent eval contamination.
5. **Sentence-embedding score** *(optional, off by default)* — LaBSE cosine ≥ 0.75 for
   web-mined sources (WikiMatrix, ParaCrawl).

### 4.3 Nested data scales

We materialize 4 strictly nested splits:

```
splits/scale_10k.parquet     ⊂  splits/scale_50k.parquet
                             ⊂  splits/scale_100k.parquet
                             ⊂  splits/scale_500k.parquet
```

Sampling: stratified mix across sources so each scale has the same source-domain
distribution (default mix: 30% News-Commentary, 25% TED2020, 25% WikiMatrix,
15% OpenSubtitles, 5% UN). Within each source, sample with a fixed RNG seed so the
smaller scales are exact prefixes of the larger ones.

Direction: each pair appears **twice** — once as `<|en2zh|>` and once as `<|zh2en|>` — so
the model learns both directions from the same data. Final per-scale row count = 2× pair
count (e.g. scale_10k → 20,000 training examples).

### 4.4 Held-out validation

A separate 5K-pair `dev.parquet` (sampled from News-Commentary + TED2020, disjoint from all
training scales) is used for validation loss / BLEU during training, regardless of training
scale. This keeps validation curves comparable across scales.

## 5. Training framework

### 5.1 Stack

| Concern | Tool |
|---|---|
| Modeling | PyTorch 2.5, 🤗 Transformers, 🤗 PEFT (LoRA opt-in only — default OFF) |
| Training loop | 🤗 TRL `SFTTrainer` |
| Distributed / mixed precision | 🤗 Accelerate + DeepSpeed ZeRO-2 (optional, for ≥1B) |
| Tokenization | SentencePiece + 🤗 Tokenizers |
| Eval | sacreBLEU, chrF++, UniEval-compatible COMET (`Unbabel/wmt22-comet-da`) |
| Tracking | Weights & Biases + MLflow (Azure ML built-in) |
| Config | YAML loaded via `omegaconf` |
| Lang ID | `fasttext-langdetect` |
| Dedup | `datasets.unique` + bloom filter for cross-source |

### 5.2 Default hyperparameters

```yaml
optimizer: adamw_torch_fused
lr: 5.0e-5            # smaller than typical SFT — new embeddings need a calm regime
betas: [0.9, 0.95]
weight_decay: 0.1
warmup_ratio: 0.03
lr_scheduler: cosine
max_seq_length: 1024
per_device_batch_size: 16
gradient_accumulation_steps: 4    # effective batch 64
epochs: 3                          # for each scale, scales independently
gradient_checkpointing: true
bf16: true                         # A100 / H100; fp16 fallback on V100
new_embedding_lr_multiplier: 10.0  # new rows get 10× LR for first 1000 steps
```

The `new_embedding_lr_multiplier` is implemented as a parameter group: rows added during
tokenizer extension are placed in a separate group with elevated LR that decays back to
base LR over the warmup window. This is the closest thing this design has to a "secret
sauce" and is the single most-likely place to need tuning.

### 5.3 Catastrophic-forgetting monitor

Every `eval_steps` we additionally measure perplexity on a fixed 5K-sentence English-only
holdout (sampled from the model's own pretraining distribution — FineWeb-Edu dev set). If
English perplexity climbs more than +20% above the base model's reference value, we log a
WARN and the run config can opt into:

- automatic LR reduction (`auto_lr_decay_on_forgetting: true`)
- early-stop the run
- restore last checkpoint

## 6. Azure resources

| Resource | Spec | Purpose | Approx cost |
|---|---|---|---|
| Azure ML Workspace | basic | experiment hub | included |
| Compute cluster | `Standard_NC24ads_A100_v4` (1× A100 80GB), min 0 / max 4 | training | ~$3.40/hr/node |
| Azure Blob (Premium LRS) | 1× container, ~200GB | datasets + checkpoints | ~$20/month for 200GB |
| Azure Key Vault | std tier | HF token, W&B key | <$1/month |
| Container Registry | basic | custom training image | ~$5/month |
| App Insights | included via AML | log aggregation | included |

The 4-run sweep at scale_500k is the heaviest run: ~500K pairs × 2 directions × 3 epochs
≈ 3M examples, at bsz 64 ≈ 47K steps. On a single A100-80GB at ~3 steps/sec ≈ 4.5 hours.
Full sweep (10K + 50K + 100K + 500K) ≈ 6-8 hours of A100 time ≈ **~$25-30 per full
sweep**.

## 7. Observability & snapshots

- **W&B project**: `english-chinese-mt` (one run per scale).
- **MLflow** runs nested under an AML experiment of the same name.
- **Checkpoints** saved every `save_steps` (default 500). Retention policy:
  - keep the **last 3 rolling** checkpoints,
  - keep every **eval-milestone-best** checkpoint permanently (best dev BLEU),
  - keep every checkpoint at **epoch boundaries** permanently.
- Checkpoints written to `models/runs/<run-id>/checkpoint-<step>/` locally, mirrored to
  Blob at `azureml://datastores/checkpoints/<run-id>/`.
- **Logs**: rotating file at `models/runs/<run-id>/train.log` + structured JSONL at
  `models/runs/<run-id>/metrics.jsonl`. Both shipped to Blob.
- **Rollback**: `python scripts/06_train.py --resume-from <checkpoint-dir>` resumes
  optimizer + scheduler + RNG state.

## 8. Deliverables

1. **Four trained models**, one per data scale (10K / 50K / 100K / 500K), each bidirectional
   (en↔zh).
2. **Checkpoint time-series per model** — every `save_steps` plus every epoch boundary —
   so we can plot a training-progress curve for each scale and roll back to any prior
   snapshot.
3. Eval report (`reports/scaling_study.md` + figures): one quality-vs-step curve per scale,
   plus a final quality-vs-data-scale summary, plus the catastrophic-forgetting curve
   overlaid.
4. Frozen tokenizer + extended embeddings published as a single HF-format model directory
   so the experiment is reproducible by any third party with the same data.

## 9. Out-of-scope (for now)

- LoRA / QLoRA — code path exists but disabled. Adding a brand new language is the worst
  case for LoRA; full fine-tune is the honest baseline.
- Reinforcement learning / preference tuning. Pure SFT only.
- Languages other than zh-Hans. Traditional Chinese / Cantonese excluded by `fasttext` lid.
- Quantization for inference. Out of scope — this is a training study.
