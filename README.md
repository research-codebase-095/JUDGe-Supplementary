# Code and cache archive

This archive accompanies the anonymized submission "Beyond Scalar Confidence:
Reliability Signals and Failure Modes in LLM-as-Judge Verdicts" (JUDGe 2026).
It contains every named script and cached tensor referenced in the paper's
Reproducibility paragraph, sufficient to reproduce every reported number from
the provided caches without rerunning model inference.

Maps every table and headline number in the paper to the exact command that
produced it and the file (or cache) that command reads/writes. Organized in
paper order. Every command below was actually run against the shipped
`data/*.pt` caches while preparing this file (Python 3.12, `pip install -r
requirements-dev.txt`); all outputs matched the paper's reported numbers
exactly, to the precision the paper reports them.

## Structure

```
scripts/     Analysis and data-collection scripts (60 files)
src/         The `deployment_reliability` package: signal extraction (Phi),
             combiner (G_w), router (R_tau), calibration, significance tests
tests/       Unit tests for src/deployment_reliability
notebooks/   Exploratory notebooks from earlier development stages
data/        Cached tensors (.pt files) - see mapping below
requirements.txt, requirements-dev.txt, pyproject.toml
```

## Setup

```bash
pip install -r requirements.txt        # or requirements-dev.txt for tests/notebooks/figures
```

All code is CPU-compatible; no CUDA/GPU dependency is required anywhere in
`src/` or `scripts/`. Every script that loads a cache does so via a relative
`data/<name>.pt` path from the repository root, so run all commands below
from the repo root.

**Not included:** raw model weights (Qwen2.5-0.5B/1.5B-Instruct, SmolLM2-360M-
Instruct, GPT-2, Pythia-160m, ResNet-50) and raw datasets (MT-Bench, WikiText-2,
ImageNet-1k, Imagenette) - all are public and downloaded automatically on
first use by the `collect_*.py` scripts, or via `scripts/download_eval_data.py`
for the vision datasets. Every number in the paper is reproducible from the
released `data/*.pt` caches alone, without rerunning any of that collection.

**Seeds:** `seed=0` fixes every split/bootstrap/reseed in this project (sweeps
0-4 where a script explicitly reseeds, e.g. `stability_across_splits.py`,
`gbt_seed_variance.py`). Re-running a `collect_judge_verdicts*.py` /
`collect_logits*.py` script performs fresh model inference and is expected to
reproduce the already-cached numbers up to floating-point noise, not
necessarily bit-for-bit; every analysis script below performs **no new
inference** and reads the shipped caches directly, so its output is exact.

**Build verification:** `pytest -q` from the repo root passes 275/275 (3
skipped pending optional large downloads: ImageNet-A/O, full ImageNet-1k val)
against a fresh `pip install -r requirements-dev.txt`.

## Section 3 (Method) - the &Phi; verdict-token restriction

| Command | Output |
|---|---|
| `python scripts/collect_judge_verdicts_2way.py` | `data/judge_feature_cache_mtbench_2way.pt`, `_1p5b_2way.pt`, `_smollm2_360m_2way.pt` (already cached; Phi restricted to the two verdict-token logits `[z_A, z_B]`, all 3 judge configs) |
| `python scripts/compare_2way_vs_fullvocab.py` | stdout: 2-way vs. full-vocabulary MSP/best-feature/combiner AUROC and MSP-discordance, side by side |

## Table 1 / Table 2 (Section 4.1) - cross-domain validation

| Command | Output |
|---|---|
| `python scripts/paper_diagnostics.py` | stdout §1 - **Table 1's exact headline row**: MSP, best single feature, combiner AUROC, comb-best diff, DeLong p, winner, for ResNet-50/GPT-2/Pythia-160m/Qwen2.5-0.5B(LLM)/Qwen2.5-0.5B(judge) |
| `python scripts/judge_characterization.py` | stdout §A/§B - Table 1/2's judge rows via question-id cluster-bootstrap CI, all 3 judge configs incl. the non-confirmatory SmolLM2-360M row |
| `python scripts/paper_ablation_and_effect_size.py` | stdout - **Table 2 exactly**: all 5 oriented single-feature AUROCs + combiner, per backbone, plus combiner-vs-MSP paired-bootstrap CI |
| `python scripts/best_feature_selected_on_cal.py` | stdout - the disjoint select-on-`threshold_cal`/report-on-`id_test` protocol Table 1 uses for "best single signal" |
| `python scripts/vision_subsample_at_judge_n.py` | stdout - the "1000 draws of vision's id_test down to n=642 reach p<0.05 in only 42%" power check |

`data/judge_feature_cache_mtbench.pt` / `_1p5b.pt` / `_smollm2_360m.pt` back
the three judge rows directly (already cached; produced by
`collect_judge_verdicts.py`, `_1p5b.py`, `_smollm2_360m.py`).
`collect_judge_verdicts_smollm2.py` (SmolLM2-**1.7B**) and
`collect_judge_verdicts_7b.py` (Qwen2.5-7B/3B) are **not** the cache the paper
uses - both were killed as infeasibly slow on the reference machine (see their
own module docstrings) and superseded by the 360M run; they back no number
here.

## Table 3 (Section 4.1) - pairwise rank-discordance

| Command | Output |
|---|---|
| `python scripts/pairwise_rank_discordance.py` | stdout, "POOLED-BOOTSTRAP-STABILIZED orientation" block - **Table 3 exactly**, all 4 configs |
| `python scripts/direction_split_robustness_check.py` | stdout - exploratory check of the per-config direction convention Table 3 uses; not itself cited by a paper number |

## Appendix A.4 - extended backbone results (GPT-2, Pythia-160m)

| Command | Output |
|---|---|
| `python scripts/paper_diagnostics.py` | stdout §3/§10 - standardized-refit AUROC (the 0.829→0.846 claim) and nonlinear-combiner AUROC, per LLM backbone |
| `python scripts/combiner_regularization_sweep.py` | stdout - L2-penalty sweep isolating whether regularization strength alone explains GPT-2's gap vs. Pythia-160m needing standardization |

## Section 4.2 / Table 7 (Appendix A.5) - Selective Judging for LLM-as-Judge Verdicts

| Command | Output |
|---|---|
| `python scripts/selective_judging.py` | stdout - base MSP/combiner routing at &tau;<sub>lo</sub>=0.500, &tau;<sub>hi</sub>=0.875 |
| `python scripts/selective_judging_calibrated_msp.py` | stdout - **Table 7 exactly**: adds temperature/Platt-scaled MSP rows, always-Execute/always-Verify baselines, 95% CIs |
| `python scripts/routing_multiplicity_check.py` | stdout - the 4-comparison Bonferroni-adjusted family (α/4=0.0125): confirms 1.5B combiner + Platt-MSP survive vs. always-verify, 0.5B does not |
| `python scripts/selective_judging_utility_and_sweep.py` | stdout Task A - reward-term robustness (−0.104→−0.084 under `u_execute_correct=1.0`); Task B - **the 20-cell cost-regime sweep** (combiner wins 5/20 valid cells at 1.5B, 0/20 at 0.5B, never loses) |

## Section 5.1 / Table 4, Table 5 (Discussion)

| Command | Output |
|---|---|
| `python scripts/equivalence_test_tost.py` | stdout - **TOST equivalence test**, both margins (0.02-AUROC convention and own-MDE sensitivity), both configs |
| `python scripts/failure_mode_table.py` | stdout - **Table 4 exactly**, all 6 cells (high-conf-wrong / low-conf-correct / high-disagreement × 0.5B/1.5B) |
| `python scripts/collect_judge_swap_consistency.py`, `scripts/collect_judge_swap_consistency_1p5b.py` | `data/judge_swap_consistency_cache.pt`, `_1p5b.pt` (already cached; explicit order-swap test, both judge sizes) |
| `python scripts/order_averaging_correction.py` | stdout - binary-verdict order-averaging (confirms it is uninformative: 1.09% swap-consistency at 0.5B, i.e. verdicts flip on 98.9% of pairs) |
| `python scripts/collect_judge_swap_consistency_continuous.py` | `data/judge_swap_consistency_cache_continuous.pt` (already cached; re-runs the 0.5B swap test, additionally persisting `phi_swapped` so a genuine continuous order-average is computable) |
| `python scripts/order_averaging_continuous_analysis.py` | stdout - **Table 5 exactly**: single-order 50.9% → two-pass order-averaged 64.0% (cluster-bootstrap 95% CI [59.0%, 68.7%], McNemar p=2.9×10⁻⁶) |
| `python scripts/verbosity_and_swap_crosstab.py` | stdout Part A - the not-a-verbosity-shortcut check (predicted-winner-matches-longer-response ≈51.9%/51.0% vs. human label's own 70.8%, McNemar p<10⁻¹¹, both configs); Part B - swap-consistency crossed with the Table 4 failure modes |

`scripts/order_averaging_continuous_analysis.py` is new in this release: it
is the missing analysis half of `collect_judge_swap_consistency_continuous.py`
(which only collects `phi_orig`/`phi_swapped`, without computing the
resulting accuracy) - added so Table 5's number has a runnable command like
every other table, rather than requiring manual reconstruction from the cache.

## Appendix A.6 - extended judge characterization

| Command | Output |
|---|---|
| `python scripts/inter_annotator_agreement.py` | stdout - 274/351=78.1% item-level agreement, 633/772=82.0% pairwise agreement |
| `python scripts/stability_across_splits.py` | stdout - 5-reseed combiner-vs-MSP stability, LLM and judge backbones |
| `python scripts/vision_seed_stability.py` | stdout - same 5-reseed treatment applied to the vision row specifically |
| `python scripts/power_analysis.py` | stdout - minimum detectable effect at 80% power: 0.036 (0.5B) / 0.018 (1.5B) |
| `python scripts/nonlinear_combiner_judge_task.py` | stdout - untuned GBT combiner AUROC/ECE vs. linear combiner, judge task |
| `python scripts/gbt_seed_variance.py` | stdout - 5-seed variance for the GBT combiner on the three LLM backbones |
| `python scripts/matched_subset_reslice.py` | stdout - re-slices Qwen's cache to SmolLM2's exact 23-question/400-row subset for an apples-to-apples comparison |
| `python scripts/question_level_resplit.py` | stdout - split at the question level (32/8/40 of 80 questions, 546/121/617 rows): diff +0.055 (p=0.161) at 0.5B, −0.007 (p=0.169) at 1.5B |
| `python scripts/calibration_reliability_bins.py` | stdout - 10-bin reliability-diagram data underlying the ECE/Brier summary (0.35/0.32→0.02/0.05) |
| `python scripts/selection_aware_bootstrap.py` | stdout - quantifies best-feature selection noise (Limitation iv), holding `id_test` fixed and resampling the calibration pool instead |

## Reproducing a headline number (example)

```python
import torch
d = torch.load("data/judge_feature_cache_mtbench_1p5b.pt", weights_only=False)
# d contains Phi feature tensors, correctness labels, and the fit/cal/test split assignment
```

`scripts/judge_characterization.py` and `scripts/paper_diagnostics.py` contain
the exact AUROC/DeLong/cluster-bootstrap machinery used throughout the paper;
every other analysis script above imports one or the other directly rather
than reimplementing it.

## Which cache backs which result

| Cache file | Backs |
|---|---|
| `judge_feature_cache_mtbench.pt` | Qwen2.5-0.5B-Instruct judge, Phi + correctness (Tables 1-4) |
| `judge_feature_cache_mtbench_1p5b.pt` | Qwen2.5-1.5B-Instruct judge, same |
| `judge_feature_cache_mtbench_2way.pt`, `_1p5b_2way.pt` | 2-way Phi restriction (Section 3, Appendix A.6) |
| `judge_feature_cache_mtbench_smollm2_360m.pt` (+`_2way`) | SmolLM2-360M non-confirmatory check (Section 4.1) |
| `judge_swap_consistency_cache.pt`, `_1p5b.pt` | Order-swap test / position-sensitivity (Section 5.1) |
| `judge_swap_consistency_cache_continuous.pt` | Continuous order-averaging (Table 5) |
| `logit_cache_resnet50.pt` | Vision backbone dev/calibration split (Imagenette) |
| `logit_cache_imagenet1k_resnet50.pt` | Full 50,000-image ImageNet-1k `id_test` (Table 1's ResNet-50 row) |
| `llm_feature_cache_gpt2.pt`, `_pythia160m.pt`, `_qwen05binstruct.pt` | LLM backbones on WikiText-2 (Table 1's LLM row, Appendix A.4) |

Everything else in `data/` (`logit_cache_vit_b16.pt`, `_convnext_tiny.pt`, the
DTD/iNaturalist/Places365/ImageNet-C variants, `mahalanobis_feature_cache_*`,
`odin_cache_*`, `cross_model_correctness_results*`, `llm_free_running_cache_*`)
supports this project's broader `deployment_reliability` pipeline at an
earlier development stage and backs no number in the current paper; included
for completeness, not individually cited above.

## Notes

- All splits, bootstraps, and reseeds fix `seed=0` (sweeps 0-4 where used).
- No file in this archive contains personally identifying information; all
  cached file-path metadata has been scrubbed to relative paths.
