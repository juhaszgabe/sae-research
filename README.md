# sae-research

**SAE features as runtime behavior monitors and steering handles in Gemma 3 IT (Gemma Scope 2).**

Research code for three questions:

1. **Detection** — from the activations of the *prompt only* (before any generated token), can we predict
   whether the model will **refuse** or **break a requested JSON format** (optionally **hedge**)? Difference-in-means,
   a linear probe on raw activations and SAE-feature probes are compared per layer under one protocol.
2. **Robustness** — does such a monitor survive distribution shift (style, source, language)?
3. **Coherence** — are the best-predicting SAE features also the best steering handles, once side effects
   (MMLU accuracy, NLL/KL) are counted?

The label is what the model *does*, not what the prompt is about, so the data deliberately contains
prompts where behavior and category disagree, and every SAE number is reported next to a baseline and
a confidence interval.

The code is a flat package (`src/`) meant to be driven from Colab notebooks or the thin CLI scripts in
`scripts/`. Every function takes explicit arguments, returns in-memory objects and writes to disk only
at the fixed locations below.

## Setup

Python 3.11 or 3.12.

```bash
pip install -e ".[dev]"
```

or, with [uv](https://docs.astral.sh/uv/) and the committed lock file:

```bash
uv sync --extra dev
```

`google/gemma-3-*` repositories are gated: accept the license once on Hugging Face and provide a token
as `HF_TOKEN`. The `google/gemma-scope-2-*` SAE repositories are not gated.

### Google Colab

```python
!git clone https://github.com/juhaszgabe/sae-research.git
%cd sae-research
!pip install -e .

import os
from google.colab import drive, userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
drive.mount("/content/drive")
os.environ["SBM_DATA_ROOT"] = "/content/drive/MyDrive/sbm/data"
os.environ["SBM_ARTIFACT_ROOT"] = "/content/drive/MyDrive/sbm/artifacts"
os.environ["HF_HOME"] = "/content/drive/MyDrive/sbm/hf"
```

Pointing `SBM_DATA_ROOT`, `SBM_ARTIFACT_ROOT` and `HF_HOME` at Drive keeps model/SAE downloads and all
outputs across disconnects; every long step resumes from what is already on disk.

## Configuration

One file, `configs/model_b.yaml`, holds every default. The model is chosen with `model.name`
(`gemma-3-270m-it` for debugging, `gemma-3-1b-it` main, `gemma-3-4b-it` optional).

Override values without editing the file:

```python
from src.config import load_config
cfg = load_config("configs/model_b.yaml", **{"model.name": "gemma-3-270m-it", "dataset.target_n": 120})
```

```bash
python scripts/extract_activations.py --behavior refusal --set model.name=gemma-3-270m-it --set dataset.target_n=120
```

All scripts accept `--config PATH`, `--behavior NAME [NAME ...]` (default: every behavior in the config)
and repeatable `--set key=value` (values parsed as YAML).

## Pipeline

Run the steps in this order. In Colab, prefix with `!`.

| # | step | script | main functions |
|---|---|---|---|
| 0 | platform pretest (PASS / WARN / FAIL report) | `scripts/pretest.py` | `activations.check_*`, `sae.load_sae`, `sae.sae_quality` |
| 1 | build prompt pools | `scripts/build_prompts.py` | `data.build_refusal_prompts`, `data.build_format_prompts` |
| 2 | generate responses and label them | `scripts/generate_and_label.py` | `data.run_generation`, `labeling.label_frame` |
| 3 | assemble, balance, split; export the manual-validation sample | `scripts/build_dataset.py --export-validation` | `data.build_dataset`, `data.export_validation_sample` |
| 4 | extract prompt-only activations | `scripts/extract_activations.py` | `activations.extract_dataset` |
| 5 | encode with the SAEs | `scripts/encode_sae.py --widths 16k 65k 262k` | `sae.encode_dataset` |
| 6 | probe sweep, comparisons, k-curve, controls | `scripts/run_probes.py` | `eval.run_probe_sweep`, `eval.compare_methods`, `eval.within_category`, `eval.k_curve`, `eval.run_controls` |
| 7 | steering sweep, side effects, trade-off, coherence | `scripts/run_steering.py` | `steering.run_steering_sweep`, `steering.run_side_effects`, `steering.tradeoff`, `steering.feature_coherence` |
| 8 | figures | `scripts/make_figures.py` | `plots.*` |

Notes on individual steps:

- **Pretest** — run on `gemma-3-270m-it` first, then on the main model. It exits non-zero on any FAIL.
- **Manual validation** — after step 3, fill the `human_label` column of `annotate.csv` with `1`, `0` or `x`
  (unclear) without looking at `key.csv`, then call `data.score_validation(validation_dir)`. It reports
  agreement and Cohen's kappa and warns below 0.7.
- **Dense baselines first** — step 6 can run before step 5 with
  `--methods dim logreg dense_topk refusal_dir category_only`; SAE methods are added later and existing
  results are kept.
- **Distribution shifts** (extension) — `build_prompts.py --pool ood --shift NAME`, then
  `generate_and_label.py --pool ood_NAME`, `build_dataset.py --shift NAME`,
  `extract_activations.py --pool ood_NAME`, `encode_sae.py --pool ood_NAME`, and
  `eval.evaluate_shift(cfg, behavior, NAME, results)` from a notebook. The Hungarian shift needs
  translations in `{data_root}/raw/translations_hu.jsonl` (`{behavior, source, base_id, text_hu}`).

## Where outputs go

`{model}` is `cfg.model.name`, `LL` a zero-padded layer index, `pool` is `main` or `ood_{shift}`.

```
{data_root}/                                   (default: data/, or $SBM_DATA_ROOT)
  prompts/{behavior}/{pool}.jsonl
  generations/{model}/{behavior}/{pool}.jsonl              # also the generation cache
  generations/{model}/{behavior}/{pool}_labels.parquet
  datasets/{model}/{behavior}/main.parquet                 # prompts + responses + labels + split + fold
  datasets/{model}/{behavior}/unused.parquet               # balanced-out rows (steering eval pool)
  datasets/{model}/{behavior}/ood_{shift}.parquet
  validation/{model}/{behavior}/annotate.csv, key.csv, report.json
  side_effects/mmlu_subset.jsonl, wikitext_passages.jsonl

{artifact_root}/                               (default: artifacts/, or $SBM_ARTIFACT_ROOT)
  activations/{model}/{behavior}/{pool}/layer_{LL}.npz, index.parquet, norms.json
  sae_codes/{model}/{behavior}/{pool}/{width}_{l0}/layer_{LL}_{position}.npz, quality.csv
  results/{model}/{behavior}/probe_results.parquet, comparisons.parquet, within_category.parquet,
          k_curve.parquet, controls.parquet, scores/, probes/, features/
  steering/{model}/{behavior}/sweep.parquet, side_effects.parquet, tradeoff.parquet,
          coherence.parquet, direction_coherence.parquet, vector_cosines.parquet, generations/
  figures/{model}/
  pretest/{model}/report.json
```

Each step also writes a `run_*.json` next to its outputs with the full config, its hash, library
versions, GPU and git commit.

## Rules the code enforces

| rule | where |
|---|---|
| Identical split, folds, metric and bootstrap for every method | `eval.evaluate_method` is the only evaluation path |
| Feature selection, scaling and hyperparameter search see only the training rows of the current fold; the test split is scored once | probes are scikit-learn estimators; `tests/test_leakage.py` |
| Prompts sharing a template or base prompt (`group_id`) never cross a split or fold | `data.make_splits`, `data.check_splits` |
| Labels come from the model's own output through deterministic labelers, with a 50-sample manual check | `labeling.py`, `data.export_validation_sample` |
| An IT model is only ever paired with IT SAEs | `config.load_config`, `sae.load_sae` |
| Steering results come with side effects and a random-direction control | `steering.make_specs`, `steering.run_side_effects` |

## Tests

```bash
pytest
```

The default run needs no network and no GPU: it uses a 4-layer random Gemma 3 model, a word-level
tokenizer with the Gemma chat format and a random SAE (`tests/conftest.py`). Tests that load the real
`gemma-3-270m-it` and its 16k SAEs are marked `hf` and are deselected by default:

```bash
pytest -m hf
```

## Known pitfalls

1. **SAE release names.** Hugging Face model cards say `…-resid_post`; SAELens calls the releases `…-res`
   (four layers, all widths) and `…-res-all` (every layer, 16k/262k, L0 small/big only).
2. **Left padding.** Plain forward passes need explicit `position_ids = cumsum(mask) − 1`, otherwise
   activations shift with the amount of padding. `model.tokenize` builds them.
3. **Double BOS.** The chat template already contains `<bos>`, so tokenize with
   `add_special_tokens=False`; `model.tokenize` raises unless there is exactly one.
4. **Layer output type.** Decoder layers return a tuple in transformers 4.x and a tensor in 5.x; the hooks
   handle both.
5. **Final-layer hidden state.** `output_hidden_states[-1]` is post-final-norm, so hooks are only compared
   against `hidden_states[L+1]` for `L < n_layers − 1`.
6. **4B model layout.** Its decoder layers live under `model.language_model.layers` (the path varies by
   version); `model.get_decoder_layers` resolves it instead of hard-coding it.
7. **YAML floats.** Write `1.0e-4`, not `1e-4`: PyYAML reads the latter as a string.
8. **Memory.** A 262k SAE is about 2.4 GB in float32 for the 1B model (5.4 GB for 4B); load one at a time.
9. **Batched generation.** If the pretest's generation-equivalence check warns, set
   `generation.batch_size: 1`.
