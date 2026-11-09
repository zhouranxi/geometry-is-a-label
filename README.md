# Geometry Is a Label, Not a Knob

Code, locked data, and self-verification for
**"Geometry Is a Label, Not a Knob: Causally Separating Representational
Dimensionality from OOD Generalization"** (under review at TMLR).

Representational dimensionality (effective dimension, D_eff) is a strong
observational predictor of out-of-distribution transfer across a
purpose-built recipe grid (r = +0.97). This repository ships the
preregistered causal battery showing that the association does **not**
support "geometry as a knob": moving D_eff by whitening does not move
transfer, noise that pins D_eff destroys it, and the formation-phase
whitening benefit is a probe-conditioning effect, not a representational
one.

## One-command verification

```bash
pip install numpy scipy
python code/verify.py
```
Re-derives **80 headline numbers** from the locked JSON files in `runs/`
and asserts each against the value printed in the manuscript — no GPU, no
downloads, a few seconds of CPU. The checks cover every experiment
(E1 operator battery, E2 architecture extension, E3 timing separation,
E4 formation-phase forcing 96-run preregistered batch + dose scan +
three-seed dose replication,
E5b dose–response mechanism adjudication, E5c conditioning measurements,
E5d MLP control, E5e saturation/headroom conjecture tests, E6 marker
generality) plus the Holm-corrected primary test family. If any check
fails, the manuscript and the data have diverged.

## Repository layout

```
code/       experiment + analysis scripts (see below)
runs/       locked per-cell JSON results (the ground truth)
results/    derived stats, bootstrap CIs, reports
paper/      TMLR LaTeX source (anonymized), compiled PDF, preregistration ADR
```

### Scripts

| script | role |
|---|---|
| `code/train_grid.py`-style training | see `runs/v2/launch_*.sh` for the exact training commands |
| `code/intervene_v2_paper.py` | E1–E3 intervention battery (needs GPU + checkpoints) |
| `code/intervene_recolor.py` | zero-noise spectral recoloring operator |
| `code/intervene_e5_dose.py` | E5b/E5d dose–response + MLP control |
| `code/intervene_e5c_cond.py` | E5c in-rank condition numbers + probe training curves |
| `code/intervene_e5d_ceiling.py`, `code/analyze_e5d_ceiling.py` | E5e saturation/headroom conjecture tests (8 paired models, orig vs whitened) |
| `code/intervene_e6_marker.py` | E6 participation ratio / effective rank |
| `code/intervene_effect_probe.py` | effect-probe pilot |
| `code/r1_id_ood.py`, `code/glue_impl.py`, `code/geo_markers.py` | shared: models, GLUE D_eff, markers |
| `code/paper3_analysis.py` | regenerates every figure + `paper3_stats.json` |
| `code/paper3_ci.py` | bootstrap CIs, permutation noise floors, Holm correction |
| `code/paper3_tables.py` | regenerates `paper/causality/tables_auto.tex` |
| `code/verify.py` | **the self-verification entry point** |
| `code/dose_shape_fit_seeds.py` | preregistered three-seed dose-scan verdict (R1-s…R4-s); `--data runs/p4_dose/dose_scan_seeds.json` runs from the shipped archive alone |
| `code/dose_shape_fit.py` | single-seed Appendix-24 shape/power-law fit (server-side provenance; reads the raw per-run directories) |
| `code/build_dose_scan_seeds.py` | folds the 27 per-run source directories into `runs/p4_dose/dose_scan_seeds.json` |
| `code/merge_p4_full.py`, `code/train_forcing_p4.py` | provenance for the 96-run forcing batch (needs GPU + checkpoints) |
| `code/make_fig5_dose.py` | regenerates `paper/causality/figs/fig5_dose.{pdf,png}` from the shipped archive (accepts either the `results/` or `runs/` layout) |
| `runs/p4_dose/dose_seeds_verdict.txt` | verbatim verdict output; the failed R1-s/R2-s criteria are reported as a limitation in the paper |

The analysis chain (`paper3_analysis.py` → `paper3_ci.py` →
`paper3_tables.py`) runs from the locked JSON alone (numpy/scipy/
matplotlib), as do `code/verify.py`, `code/dose_shape_fit_seeds.py --data`,
and `code/make_fig5_dose.py`. Scripts that address raw per-run directories
or training (`train_forcing_p4.py`, `merge_p4_full.py`,
`dose_shape_fit.py`, `build_dose_scan_seeds.py`) are shipped as provenance:
they retain the original server paths and are **not** needed to re-derive
any number in the manuscript. The intervention experiments additionally
need a GPU, CIFAR-10/100, and the trained checkpoints (not redistributed
here).

## Key numbers (all asserted by `code/verify.py`)

- Observational anchor: r(D_eff, c100) = +0.97, pooled slope +0.137 pp/unit
- Recolor-only geometric causal share: **7.7%** of the observational slope
  (bootstrap 95% CI [0.9%, 46%]; permutation noise floor [−6.9, +6.8]%)
- Anisotropic noise at pinned D_eff: −10.8 pp median transfer harm (8/8)
- Spectral rotation (exact spectrum preservation): max |Δacc| = 1.7 pp
- E3 formation gain +1.7 pp (7/8, Holm p = 0.016) vs. inference-arm harm
  −0.13 pp (7/8, Holm p = 0.016)
- E5b: probe-space arm alone reproduces the entire dose–response
  (+1.9/+3.1/+3.0 pp); formation-readout arm shows no benefit at any dose
  (+0.11 to −11.5 pp, harm growing with dose); placebo destroys head
  training in 7/9 cells
- E5c: whitening reduces the in-rank condition number by up to ~3.3×10⁵
  and lowers reachable loss (3.43 → 3.01) while slowing early epochs
- E5e: the two E3 stratification conjectures are now measurements —
  *operator saturation* survives in sharpened form (the negative model's
  covariance is already nearly whitened: in-rank cond 294, whitening
  improves it only 22× vs a 4.6×10³× median, reachable loss does not
  drop, benefit +0.08 pp), while *high-D headroom* is **refuted**
  (Spearman ρ(d_eff, benefit) = +0.857, p = 0.007; high-d_eff models
  average +3.5 pp vs +0.9 pp). E5c overlap: cond exact 15/18, 18/18
  within 1e-6, acc 18/18
- E6: PR/erank anchor observationally (+0.77/+0.86) but are *raised* ~30%
  by transfer-destroying noise — every spectrum marker is a recipe
  fingerprint, none is a knob
- E4: 96 preregistered formation-phase forcing runs move D_eff without
  moving transfer (39/39 arms same sign, 34/39 at or below the
  preregistered label threshold); paired whiten@λ=0.1 vs placebo
  +0.030 (t = 6.89, d = 1.44) implies a net slope of 14.7% of the
  observational slope. A five-point dose scan is monotone, sub-linear
  and saturating; the single-seed power-law exponent b = 0.502 does
  **not** replicate across seeds (0.502 / 0.325 / 0.264), nor does the
  low-dose cliff. The low-D recipe is inert on both replication seeds
  (10/10 inside the ±0.022 placebo band), and exactly 1 of 30 dose
  points loses function. The 14.7% is a batch-level mean, not a
  per-intervention ceiling: five of the 39 arms exceed the 20% label
  threshold (max 54.5%), and a post-hoc secant recheck of the dose
  archive finds two replication-seed low-dose points at 30–33% of
  observational (Limitations (vi) of the manuscript). These failed
  preregistered criteria are reported as limitations in
  `paper/causality/main.pdf`.

## Preregistration

`paper/adr_0016.md` contains the preregistered design (Appendices 4–5),
the dose–response falsification criteria with amendment 1 (Appendix 8),
and the full decision log (Appendices 9–10: three rounds of adversarial
self-audit, ~40 corrections, none changing the direction of any
conclusion; Appendix 14: the E5e conjecture tests, whose audit caught
three overstated sentences before they reached the compiled PDF).
Appendices 16–20 freeze the 96-run forcing battery and its verdict before
the data existed; Appendices 23–24 freeze and adjudicate the five-point
dose scan; Appendix 25 freezes R1-s…R4-s for the three-seed replication
and records, after all 12 runs landed, that R1-s (exponent replication)
and R2-s (cliff replication) **failed** while R3-s (low-D inertness)
replicated and R4-s found 1/30 function-loss points. The manuscript was
downgraded accordingly; the criteria were never revised after the data.

## Data

All numbers derive from 8 purpose-trained ResNet-18 recipes (3 lr × wd
grids × seeds) × 19 intervention conditions = 152 cells, plus
architecture-extension cohorts (ResNet-34/50, ShuffleNet-V2-x1.0,
RegNetY-400MF, 4 recipes per family). E4 adds `runs/p4_full/` (the 96-run
forcing batch and its three base readouts) and `runs/p4_dose/`: a
single-seed five-point scan plus `dose_scan_seeds.json`, a self-contained
archive of 36 base readouts and 30 dose points covering three seeds × two
recipes × five λ. The 12 newly trained replication runs are folded into
that JSON, so no per-run server directory is needed to reproduce the
verdict. Checkpoints (~2.8 GB) are available from the authors on request.

## Citation

If you use this archive, please cite it:

- Archive: Ranxi Zhou. *Geometry Is a Label, Not a Knob: Causally
  Separating Representational Dimensionality from OOD Generalization*
  (code, data, and self-verification), Zenodo, v1.2.0 working tree,
  concept DOI [10.5281/zenodo.23967259](https://doi.org/10.5281/zenodo.23967259)
  (latest published release: v1.1.0,
  [10.5281/zenodo.23993000](https://doi.org/10.5281/zenodo.23993000);
  v1.0.0: [10.5281/zenodo.23967260](https://doi.org/10.5281/zenodo.23967260)).
  The v1.2.0 tree (E4 forcing, three-seed dose replication, 80 checks) is
  the working snapshot accompanying the TMLR submission; it has not been
  deposited as a new Zenodo version.
- Manuscript: under review at Transactions on Machine Learning Research
  (see `paper/causality/main.pdf`).

## License

MIT. CIFAR-10/100 are not redistributed; the experiments download them
via torchvision.
