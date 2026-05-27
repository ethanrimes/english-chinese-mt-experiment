# FLORES-200 zh↔en Benchmarks & Expected Scores

Reference document for the SmolLM2-360M en↔zh fine-tuning scaling study.
Setup recap: 360M decoder-only LM, English-only pretraining (FineWeb-Edu, zero
Chinese contamination), tokenizer extended with ~16K randomly initialized
Chinese BPE pieces, bidirectional SFT only. Eval = FLORES-200 devtest, BLEU
(sacreBLEU), chrF++, COMET (`Unbabel/wmt22-comet-da`).

> **Honesty disclaimer.** No published paper trains a 360M English-only LM
> from scratch on Chinese via tokenizer extension + parallel-data SFT alone.
> Every number below is either (a) a published reference point for a
> *different* setup or (b) an extrapolation. Treat the "expected" column as a
> hypothesis the experiment will refute or confirm, not a target.

---

## TL;DR — expected scores per data scale

The table below gives plausible **ranges** for our setup. Anchors:
top-quartile NMT-from-scratch scaling curves on Chinese-English suggest BLEU
grows roughly log-linearly with parallel-data size between ~10K and ~10M pairs
(Koehn & Knowles 2017 — NMT < SMT under ~100M words on en-es; the curve is
similar for zh-en). We lean ~3-6 BLEU pessimistic vs published from-scratch
transformer baselines because our base model is (i) tiny, (ii) had no Chinese
in pretraining, and (iii) is decoder-only (no encoder bias).

| Pairs | BLEU zh→en | chrF++ zh→en | COMET zh→en | BLEU en→zh | chrF++ en→zh | COMET en→zh | Rough qualitative |
|---|---|---|---|---|---|---|---|
| 10K   | 1–3   | 10–18 | 0.40–0.50 | 0–2  | 5–12  | 0.35–0.45 | Mostly broken; copies, code-switching, hallucinations. Tokenizer-init noise dominates. |
| 50K   | 3–7   | 18–26 | 0.50–0.60 | 2–6  | 10–18 | 0.45–0.55 | Recognisable but ungrammatical; lexical match only. |
| 100K  | 6–11  | 24–32 | 0.55–0.65 | 4–10 | 15–24 | 0.50–0.60 | Coherent short sentences; long sentences degrade hard. |
| 500K  | 12–17 | 32–42 | 0.65–0.74 | 9–16 | 22–32 | 0.60–0.72 | Useable for gist translation; persistent errors on idioms / NEs. |
| 1M    | 15–20 | 38–47 | 0.70–0.78 | 13–20| 26–36 | 0.68–0.77 | Roughly mBART-50-bilingual-finetune territory. |
| 5M    | 19–25 | 44–53 | 0.76–0.82 | 17–25| 30–42 | 0.74–0.82 | Approaches NLLB-distilled-600M for zh→en; en→zh still weak (decode-side). |

**Asymmetry warning.** en→zh is consistently harder than zh→en in published
results across NLLB, ALMA, Tower, GemmaX2 — typically by 4–10 BLEU. With a
freshly-initialized Chinese output vocabulary the asymmetry will be *more
extreme* than literature reports; expect en→zh to lag by 6–12 BLEU until at
least the 1M scale.

Citation basis: Koehn & Knowles 2017 (NMT scaling), ALMA paper (low-data SFT
scaling, [arXiv 2309.11674](https://arxiv.org/abs/2309.11674)), "Quality or
Quantity?" ([arXiv 2408.12780](https://arxiv.org/abs/2408.12780)) which
documents an early plateau for LLM-MT SFT.

---

## Baselines to beat (FLORES-200 devtest, zh↔en)

Mixed sources. Where the original paper does not report FLORES-200 devtest
explicitly we substitute the closest published benchmark and flag it. Numbers
are sacreBLEU unless stated; spBLEU is ~2–3 points higher on zh.

| System | Size | zh→en BLEU | en→zh BLEU | zh→en chrF / COMET | en→zh chrF / COMET | Source / notes |
|---|---|---|---|---|---|---|
| **Helsinki-NLP/opus-mt-zh-en** | ~75M | 36.1 (Tatoeba-test) | — | chrF 0.548 | — | [HF model card](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en); Tatoeba is much easier than FLORES; on FLORES expect ~15–18 BLEU. |
| **Helsinki-NLP/opus-mt-en-zh** | ~75M | — | 31.4 (Tatoeba-test) | — | chrF 0.268 | [HF model card](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh); on FLORES expect ~22–28 BLEU. |
| **mBART-50 (1-to-many / many-to-many fine-tuned)** | 610M | ~16–20 | ~17–22 | — | — | [Tang et al. 2020](https://arxiv.org/pdf/2008.00401); multilingual fine-tune, FLORES range from follow-on work. |
| **NLLB-200 distilled-600M** | 600M | ~25–28 spBLEU ≈ ~22–25 BLEU | ~20–23 spBLEU ≈ ~17–20 BLEU | — / ~0.80 COMET | — / ~0.78 COMET | [NLLB paper](https://arxiv.org/abs/2207.04672); [HF model card](https://huggingface.co/facebook/nllb-200-distilled-600M). |
| **NLLB-200 dense 1.3B** | 1.3B | ~26–29 BLEU | ~21–24 BLEU | — | — | NLLB paper; +~2 spBLEU over distilled-600M. |
| **NLLB-200 dense 3.3B** | 3.3B | ~28–31 BLEU | ~24–27 BLEU | — | — | [HF model card](https://huggingface.co/facebook/nllb-200-3.3B). |
| **NLLB-200 MoE-54B** | 54B | ~31–33 BLEU | ~27–30 BLEU | — | — | NLLB paper Table 39 vicinity; SoTA among Meta's NLLB family. |
| **ALMA-7B-LoRA** | 7B | ~24 BLEU / 80.3 COMET-22 (WMT22) | ~28 BLEU / 84.9 COMET-22 (WMT22) | — | — | [ALMA paper](https://arxiv.org/abs/2309.11674); 58K high-quality pairs in stage 2 after 20B-token monolingual stage 1. |
| **ALMA-13B-LoRA** | 13B | ~25 BLEU / 80.8 COMET-22 | ~39.8 BLEU / 86.0 COMET-22 | — | — | ALMA paper. |
| **TowerInstruct-7B v0.2** | 7B | competitive with GPT-3.5 | competitive with GPT-3.5 | — | — | [Tower paper](https://arxiv.org/abs/2402.17733); built on Llama-2 + 20B continued pretrain on multilingual+code. |
| **BigTranslate (Llama-13B + 102 langs)** | 13B | ~21 BLEU FLORES | ~28 BLEU FLORES | — | — | [BigTranslate paper](https://arxiv.org/abs/2305.18098). |
| **GemmaX2-28-9B** | 9B | **45.07 spBLEU / 88.95 COMET** | **39.72 spBLEU / 88.35 COMET** | — | — | [arXiv 2502.02481](https://arxiv.org/html/2502.02481v2) FLORES-200 devtest. |
| **Gemma2-9B base (zero-shot)** | 9B | 42.00 spBLEU / 87.94 COMET | 33.05 spBLEU / 84.65 COMET | — | — | Same paper. |
| **GPT-4 (zero-shot)** | ? | ~28.5 BLEU FLORES | strong, ≈ Google | — | — | [Jiao et al. 2023](https://arxiv.org/pdf/2301.08745) ChatGPT-as-translator. |
| **Google Translate** | ? | ~31.7 BLEU FLORES | ~38 BLEU FLORES | — | — | Same paper; commercial ceiling. |
| **WMT22 zh→en winner (constrained)** | — | **33.5 sacreBLEU (newstest22)** | — | — | — | Vega-MT, [arXiv 2209.09444](https://arxiv.org/abs/2209.09444); domain is news not FLORES but useful ceiling. |

**Realistic comparable baseline for us:** NLLB-200-distilled-600M. It is the
closest in size (600M ≈ 360M base + ~50M new embeddings/LM-head for 16K new
pieces), trained from scratch as a dedicated MT model. If our 5M-pair model
gets within ~5 BLEU of distilled-600M on zh→en, that is a strong result given
zero Chinese in pretraining.

---

## Catastrophic-forgetting watch points

What the literature says about acceptable English degradation:

- **"LLaMA Beyond English"** ([arXiv 2401.01055](https://arxiv.org/html/2401.01055v2))
  shows English perplexity exploding from **14.7 → 198.8** when training
  exclusively on Chinese for 1M samples. Mixing English back in (multilingual
  joint training) keeps both PPLs low. Rule of thumb: any factor >2x English
  PPL increase is a red flag.
- **"Emergent Abilities under Continued Pretraining"**
  ([arXiv 2506.00288](https://arxiv.org/html/2506.00288v1)) finds that without
  English data in the mix, in-context-learning ability "plummets to nearly zero
  in the first few steps" — and downstream metrics only lag this by ~3000
  steps. Validation perplexity is **not** a sufficient warning signal:
  English-free and English-mixed runs hit similar PPL but very different
  downstream scores. **Implication: do not rely on English PPL alone; also
  track an English downstream task (e.g. HellaSwag, ARC-easy, or a short
  generation-quality probe).**
- The same paper documents parameter-space divergence: without English,
  parameters drift **7x further by step 100, 15x further by step 1000** than
  the English-mixed run. They mitigate via curriculum (English in first 10% of
  steps) or EMA regularization.
- Cui et al. **Chinese-LLaMA** ([arXiv 2304.08177](https://arxiv.org/html/2304.08177v3))
  added 20K Chinese tokens to a 32K LLaMA vocabulary and continued-pretrained
  on 20–120GB Chinese corpus. They do not report English degradation
  quantitatively, but their recipe (LoRA on embeddings + LM head + transformer)
  is designed to limit it.

**Concrete thresholds for our run:**
- English PPL increase **<1.5x** on a held-out WikiText-style English set: fine.
- **1.5x–3x**: tolerable for a translation-only goal but lose general English ability.
- **>3x**: forgetting; mix more English into batches, or freeze base layers, or LoRA.

Because we're doing **bidirectional** training with English on both sides of
the pair (input or output in every batch), the catastrophic-forgetting risk is
much lower than mono-direction Chinese-only continued pretraining. We expect
PPL to stay within 1.5x.

---

## Quality "ceilings" per data scale — what the literature suggests

1. **10K pairs.** ALMA's stage-2 fine-tune uses only ~58K *high-quality* pairs
   after a 20B-token monolingual stage 1, and reports the bulk of zero-shot
   improvements come from this stage (ALMA paper, [arXiv 2309.11674](https://arxiv.org/abs/2309.11674)).
   But ALMA leverages an already-multilingual Llama-2; we start at zero. At
   10K with random Chinese embeddings, expect mostly broken output. This is
   the regime where prompts/few-shot fail outright for new languages.
2. **50K pairs.** Le Scao & Rush (["How Many Data Points Is a Prompt Worth?"](https://aclanthology.org/2021.naacl-main.208/))
   showed a single prompt is worth a few hundred to a few thousand examples on
   English classification; for cross-lingual generation with randomly init'd
   embeddings, the breakeven for "prompt vs zero-shot" is much higher. 50K is
   enough for the embeddings to converge somewhat but not for fluent decoding.
3. **100K pairs.** "Quality or Quantity?" ([arXiv 2408.12780](https://arxiv.org/abs/2408.12780))
   argues LLM-MT performance plateaus *early* on the SFT data axis when the
   base LM is already multilingual. We are not — so we should *not* see early
   plateau here. This is where embedding training should start dominating.
4. **500K pairs.** Comparable in scale to standard NMT bilingual baselines
   from the WMT 2017–2019 era. A from-scratch transformer-base on WMT zh-en
   with ~500K filtered pairs typically reaches BLEU 15–20 on newstest. We
   should land in that ballpark.
5. **1M pairs.** mBART-50 bilingual fine-tune territory ([Tang et al. 2020](https://arxiv.org/pdf/2008.00401)).
   This is also the scale of TED-talks-only systems.
6. **5M pairs.** Approaching the data scale where NLLB-distilled-600M was
   trained (NLLB's effective per-pair training is much larger due to
   many-to-many transfer, but 5M raw pairs is a meaningful threshold). Expect
   the rate of BLEU growth per doubling to flatten visibly. Beyond ~10M pairs
   model capacity (we have only 360M params) starts to bind.

---

## Key papers — one-line takeaways

- **NLLB-200 — No Language Left Behind** ([arXiv 2207.04672](https://arxiv.org/abs/2207.04672)) — primary FLORES-200 reference. 600M-distilled is the most-comparable-size baseline.
- **mBART-50** ([Tang et al. 2020, arXiv 2008.00401](https://arxiv.org/pdf/2008.00401)) — random-init token embeddings extension on a multilingual base; demonstrates that adding language tokens with random embeddings works.
- **Chinese-LLaMA / Cui et al.** ([arXiv 2304.08177](https://arxiv.org/html/2304.08177v3)) — the canonical tokenizer-extension recipe: SentencePiece on Chinese, append 20K tokens, resize embeddings + LM head, continued-pretrain.
- **LLaMA Beyond English** ([arXiv 2401.01055](https://arxiv.org/html/2401.01055v2)) — 0.5B tokens of Chinese can outperform 30B if vocabulary is right; documents English PPL explosion.
- **ALMA / ALMA-R** ([arXiv 2309.11674](https://arxiv.org/abs/2309.11674)) — two-stage recipe (monolingual stage 1, ~58K parallel stage 2). Stage 2 alone gives +16 BLEU. Suggests our high-data scales may saturate without back-translation.
- **X-ALMA** ([arXiv 2410.03115](https://arxiv.org/pdf/2410.03115)) — modular per-language experts; SoTA across 50 langs vs Aya-101/23 on FLORES-200.
- **BigTranslate** ([arXiv 2305.18098](https://arxiv.org/abs/2305.18098)) — Llama-13B + 102-lang parallel; #4 on FLORES-200 BLEU at release. Confirms LLM-based MT can be competitive even without an MT-purpose architecture.
- **TowerLM / Tower-Instruct** ([arXiv 2402.17733](https://arxiv.org/abs/2402.17733)) — Llama-2 + 20B continued pretrain + instruct. Tower-7B ≈ GPT-3.5 on MT.
- **GemmaX2-28** ([arXiv 2502.02481](https://arxiv.org/html/2502.02481v2)) — 9B Gemma2 reaches 45 spBLEU / 88.95 COMET zh→en on FLORES-200, the best small-open ceiling we found.
- **Quality or Quantity?** ([arXiv 2408.12780](https://arxiv.org/abs/2408.12780)) — for LLM-MT, SFT data scale plateaus early; diversity hurts more than it helps. Our zero-Chinese setting probably *doesn't* plateau as early.
- **How Many Data Points Is a Prompt Worth?** ([Le Scao & Rush 2021](https://aclanthology.org/2021.naacl-main.208/)) — useful conceptual frame: a single prompt is worth ~100s–1000s of training examples on familiar tasks; expect ~zero benefit for new-script languages.
- **Koehn & Knowles 2017 — Six Challenges for NMT** — classic; NMT underperforms SMT under ~100M source words. Anchors low-data expectations.
- **WMT22 General MT — Vega-MT** ([arXiv 2209.09444](https://arxiv.org/abs/2209.09444)) — SoTA on newstest22 zh→en at 33.5 sacreBLEU; news-domain ceiling.

---

## Risks — why our numbers could be *worse* than literature suggests

1. **Model size.** Most published "small" MT models are 600M–1.3B. At 360M we
   have ~half the parameters of NLLB-200-distilled-600M and we are
   decoder-only (no encoder inductive bias). Expect a ~3–6 BLEU penalty vs
   same-data encoder-decoder baselines.
2. **Zero Chinese contamination.** Unlike Llama-2, mBART, or even FineWeb-EN
   models that have residual web Chinese, FineWeb-Edu is aggressively
   filtered. **All** Chinese token embeddings start random. Most prior
   "tokenizer extension" papers start from a base that has at least seen
   *some* of the target language.
3. **Bidirectional combined training.** Helpful for catastrophic forgetting,
   but the model splits capacity between two directions. Expect each direction
   to be ~1–3 BLEU below a same-data unidirectional system.
4. **Decoder-only architecture.** Optimal NMT systems are encoder-decoder.
   Decoder-only LMs need 2–4× more parameters to match encoder-decoder MT
   quality at the same FLOPS budget (folklore + Tower paper observations).
5. **No back-translation, no monolingual data, no iterative refinement.** SFT
   only. NLLB-distilled-600M was trained with extensive back-translation and
   self-distillation. Without these, expect ~3–5 BLEU below NLLB at equal
   parallel-pair budget.
6. **COMET is unforgiving of hallucinations.** A small model trained on
   noisy parallel data often produces fluent-but-wrong outputs; COMET drops
   harder than BLEU in that regime. Our COMET ranges above may be optimistic
   if data quality is mediocre.

**Net guidance.** If our 5M-pair model lands at BLEU ≥18 / COMET ≥0.74 on
zh→en FLORES-200 devtest, that is publishable as a positive result for the
"can a tiny English-only LM learn Chinese MT from scratch?" question. If it
lands at BLEU ≥22 / COMET ≥0.78 (matching NLLB-distilled-600M within 3 BLEU)
that is a strong result. Anything above NLLB-distilled-600M would be
surprising and should trigger leak / contamination checks on the training
data vs FLORES-200 devtest.

---

## Sources

- [NLLB paper (arXiv 2207.04672)](https://arxiv.org/abs/2207.04672) — Nature version: [doi:10.1038/s41586-024-07335-x](https://www.nature.com/articles/s41586-024-07335-x).
- [ALMA (arXiv 2309.11674)](https://arxiv.org/abs/2309.11674); [X-ALMA (arXiv 2410.03115)](https://arxiv.org/pdf/2410.03115).
- [Chinese-LLaMA (arXiv 2304.08177)](https://arxiv.org/html/2304.08177v3).
- [LLaMA Beyond English (arXiv 2401.01055)](https://arxiv.org/html/2401.01055v2).
- [Emergent Abilities under CPT (arXiv 2506.00288)](https://arxiv.org/html/2506.00288v1).
- [GemmaX2 / Open LLM MT empirical (arXiv 2502.02481)](https://arxiv.org/html/2502.02481v2).
- [BigTranslate (arXiv 2305.18098)](https://arxiv.org/abs/2305.18098).
- [TowerLM (arXiv 2402.17733)](https://arxiv.org/abs/2402.17733).
- [Quality or Quantity? (arXiv 2408.12780)](https://arxiv.org/abs/2408.12780).
- [Le Scao & Rush — How Many Data Points Is a Prompt Worth? (NAACL 2021)](https://aclanthology.org/2021.naacl-main.208/).
- [Vega-MT / WMT22 winner (arXiv 2209.09444)](https://arxiv.org/abs/2209.09444).
- [mBART-50 / Multilingual Translation w/ Extensible Pretraining (arXiv 2008.00401)](https://arxiv.org/pdf/2008.00401).
- [Helsinki-NLP/opus-mt-zh-en (HuggingFace)](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en).
- [Helsinki-NLP/opus-mt-en-zh (HuggingFace)](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh).
- [facebook/nllb-200-distilled-600M (HuggingFace)](https://huggingface.co/facebook/nllb-200-distilled-600M).
- [facebook/nllb-200-3.3B (HuggingFace)](https://huggingface.co/facebook/nllb-200-3.3B).
- [Jiao et al. — Is ChatGPT A Good Translator? (arXiv 2301.08745)](https://arxiv.org/pdf/2301.08745).
- [Helsinki-NLP OPUS-MT-leaderboard (GitHub)](https://github.com/Helsinki-NLP/OPUS-MT-leaderboard).
