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

Re-derives **42 headline numbers** from the locked JSON files in `runs/`
and asserts each against the value printed in the manuscript — no GPU, no
downloads, a few seconds of CPU. The checks cover every experiment
(E1 operator battery, E2 architecture extension, E3 timing separation,
E5b dose–response mechanism adjudication, E5c conditioning measurements,
E5d MLP control, E6 marker generality) plus the Holm-corrected primary
test family. If any check fails, the manuscript and the data have
diverged.

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
| `code/intervene_e6_marker.py` | E6 participation ratio / effective rank |
| `code/intervene_effect_probe.py` | effect-probe pilot |
| `code/r1_id_ood.py`, `code/glue_impl.py`, `code/geo_markers.py` | shared: models, GLUE D_eff, markers |
| `code/paper3_analysis.py` | regenerates every figure + `paper3_stats.json` |
| `code/paper3_ci.py` | bootstrap CIs, permutation noise floors, Holm correction |
| `code/paper3_tables.py` | regenerates `paper/causality/tables_auto.tex` |
| `code/verify.py` | **the self-verification entry point** |

The analysis chain (`paper3_analysis.py` → `paper3_ci.py` →
`paper3_tables.py`) runs from the locked JSON alone (numpy/scipy/
matplotlib). The intervention experiments additionally need a GPU,
CIFAR-10/100, and the trained checkpoints (not redistributed here).

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
- E6: PR/erank anchor observationally (+0.77/+0.86) but are *raised* ~30%
  by transfer-destroying noise — every spectrum marker is a recipe
  fingerprint, none is a knob

## Preregistration

`paper/adr_0016.md` contains the preregistered design (Appendices 4–5),
the dose–response falsification criteria with amendment 1 (Appendix 8),
and the full decision log (Appendices 9–10: three rounds of adversarial
self-audit, ~40 corrections, none changing the direction of any
conclusion).

## Data

All numbers derive from 8 purpose-trained ResNet-18 recipes (3 lr × wd
grids × seeds) × 19 intervention conditions = 152 cells, plus
architecture-extension cohorts (ResNet-34/50, ShuffleNet-V2-x1.0,
RegNetY-400MF, 4 recipes per family). Checkpoints (~2.8 GB) are available
from the authors on request.

## License

MIT. CIFAR-10/100 are not redistributed; the experiments download them
via torchvision.
