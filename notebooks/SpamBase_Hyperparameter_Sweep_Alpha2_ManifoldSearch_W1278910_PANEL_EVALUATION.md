# Panel Evaluation: `SpamBase_Hyperparameter_Sweep_Alpha2_ManifoldSearch_W1278910.ipynb`

## Scope
This note evaluates the panel families generated in the notebook's final analysis section:

- Per-W scatter panels: `validation/test accuracy vs alpha_primary_mean`.
- Per-W faceted sweep panels for validation and test accuracy.
- Per-W `validation accuracy vs alpha distance to 2`.
- Cross-sweep heatmaps for `Spearman(alpha, val_acc)` and `Spearman(alpha, test_acc)`.
- Cross-W bar charts for best competitive alpha and best validation accuracy.
- Cross-W alpha-comparison scatterplots (`W7 vs W8`, `W8 vs W9`, `W8 vs W10`).

## Executive readout
1. **Global ceiling is nearly tied across W variants** on validation (`~0.966`) and very close on test (`~0.9522-0.9535`), so W mostly changes the **alpha operating range** rather than the achievable top accuracy.
2. **W10 is heterogeneous by sweep**: some sweeps are alpha-helpful (`max_depth`, `min_child_weight`), while others show reverse test trends (`eta_x_n_estimators`, `reg_lambda`, weakly `subsample`).
3. The notebook's own sweep classifier labels W10 as a mix of `alpha_helpful`, `capacity_tradeoff`, and `inconclusive`, matching your observation that reverse trends appear in a substantial portion of sweeps.
4. The best low-alpha choices lose little validation accuracy and can improve test accuracy slightly versus the best-val picks, indicating a robust tradeoff region just above alpha 2.

## Panel-by-panel interpretation

### 1) Per-W scatter panels (`val/test accuracy vs alpha_primary_mean`)
- Across W, the dominant pattern is **rising then flattening** accuracy as alpha increases (diminishing returns).
- W determines the alpha scale where this happens:
  - Lower-alpha regime: W1/W10.
  - Mid-alpha regime: W2/W8/W9.
  - High-alpha regime: W7.
- Interpretation: W behaves like a representation-dependent shift in alpha calibration more than a change in maximum attainable accuracy.

### 2) Per-W faceted sweep panels (validation + test)
- These panels are the most diagnostic for confounding/Simpson effects because they hold sweep context fixed.
- W10 specifically shows mixed slope signs:
  - Positive/alpha-helpful: `max_depth`, `min_child_weight`.
  - Negative on test (capacity-tradeoff behavior): `eta_x_n_estimators`, `reg_lambda` (and near-flat/negative `subsample`).
- Practical implication: for W10, alpha should be tuned **jointly with sweep family**, not with a single global monotonic rule.

### 3) `val accuracy vs alpha distance to 2`
- These panels should show a local optimum or plateau near the hard/soft alpha bands rather than strict monotonic behavior.
- Given the final model table, the competitive solutions cluster near alpha slightly above 2 for most W (except W7), consistent with a near-target operating zone.

### 4) Heatmaps: Spearman correlations by sweep
- The heatmaps summarize the faceted behavior:
  - Positive correlation in `max_depth` and `min_child_weight`.
  - Negative correlation in `eta_x_n_estimators` (and often test-side `reg_lambda`/`subsample` behavior).
- For W10 this is not noise-only: the slopes are directionally consistent with the classification labels.

### 5) Bar chart: best competitive alpha by matrix
- Ordering indicates W-specific alpha calibration:
  - Lowest competitive alphas near W10/W1,
  - highest for W7,
  - others in between.
- Interpretation: if alpha budget matters, W10/W1 are operationally attractive.

### 6) Bar chart: best validation accuracy by matrix
- Bars are effectively tied at the top across W, reinforcing that representation mostly re-parameterizes alpha rather than changing top-end accuracy.

### 7) Cross-W alpha comparison scatterplots
- These support the scale-shift story: same configuration IDs map to different alpha ranges depending on W.
- `W8 vs W10` and `W8 vs W9` should be tighter than `W7 vs W8`, with W7 generally shifted higher.

## Numeric highlights from the notebook tables

### Alpha range summary (by W)
- W1: alpha range `1.788 - 2.574`
- W10: alpha range `1.750 - 2.386`
- W2: alpha range `2.099 - 2.845`
- W7: alpha range `2.520 - 3.831`
- W8: alpha range `2.119 - 3.074`
- W9: alpha range `2.317 - 3.165`

### Final model comparison (best-val vs best-low-alpha)
- W10:
  - best val: `val=0.966033`, `test=0.952226`, `alpha=2.332814`
  - best low-alpha: `val=0.964130`, `test=0.953529`, `alpha=2.230249`
- Same pattern (small val drop, similar/slightly better test) appears for W1/W2; W8/W9 are essentially tied between selections.

### W10 sweep-level signs (from diagnostics table)
- `eta_x_n_estimators`: negative test slope (`-0.009279`) -> `capacity_tradeoff`
- `reg_lambda`: negative test slope (`-0.006433`) -> `capacity_tradeoff`
- `subsample`: near-zero/negative test slope (`-0.002546`) -> `inconclusive`
- `max_depth`: positive test slope (`+0.016713`) -> `alpha_helpful`
- `min_child_weight`: positive test slope (`+0.019930`) -> `alpha_helpful`

## Recommendation
- Treat W10 as **interaction-heavy**: alpha is not globally monotonic across all sweep families.
- Choose model candidates with sweep-conditional criteria (per-sweep knee/plateau), then compare finalists globally.
- If you want a single conservative operating point, prefer low-alpha competitive models around `alpha ~2.2-2.3` for W10, where generalization is strong and validation loss vs best-val is small.
