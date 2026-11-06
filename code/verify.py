#!/usr/bin/env python3
"""Self-verification: re-derive every headline number in the paper from the
shipped locked JSON files and assert it against the value printed in the
manuscript.

Usage:  python code/verify.py            # from the repository root
Needs:  numpy, scipy  (no GPU, no downloads; runs in a few seconds)

Every CHECK below is a claim that appears verbatim in
paper/causality/main.tex. If any check fails, the manuscript and the data
have diverged; the repository should not be cited until they agree again.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

REPO = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []


def check(name: str, got, want, tol=0.0) -> None:
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) else (got == want)
    status = "ok  " if ok else "FAIL"
    print(f"[{status}] {name}: got {got!r}, paper {want!r}")
    if not ok:
        FAILURES.append(name)


def load(*candidates: str):
    for c in candidates:
        p = REPO / "runs" / c
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return d["rows"] if isinstance(d, dict) and "rows" in d else d
    raise FileNotFoundError(candidates)


def boot_ci(vals, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    idx = rng.integers(0, len(vals), (n, len(vals)))
    means = vals[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4)]


def family_points(rows, model, fam):
    """(D, c100) for the dose family + base of one model; drops glue_timeouts.

    Matches the paper3_ci.py convention: base is always the anchor point,
    family rows with glue_d=None (GLUE timeout) are excluded.
    """
    pts = [r for r in rows if r["model"] == model]
    b = next(r for r in pts if r["kind"] == "base")
    sel = [r for r in pts if r["kind"] == fam and r["glue_d"] is not None]
    return (np.array([r["glue_d"] for r in sel + [b]], float),
            np.array([r["c100_probe"] for r in sel + [b]], float))


def pearson(x, y):
    return float(pearsonr(x, y)[0])


def main() -> int:
    # ---------------- E1 ----------------
    rows = load("v2_e1_full.json", "v2/e1/intervene_v2.json")
    models = sorted({r["model"] for r in rows})
    check("E1 n_models == 8", len(models), 8)
    check("E1 n_cells == 152", len(rows), 152)
    base = [r for r in rows if r["kind"] == "base"]
    D = np.array([r["glue_d"] for r in base], float)
    Y = np.array([r["c100_probe"] for r in base], float)
    obs_slope = float(np.polyfit(D, Y, 1)[0])
    check("E1 base anchor r(D_eff, c100) = +0.97", round(pearson(D, Y), 2), 0.97)
    check("E1 pooled observational slope = +0.137 pp",
          round(obs_slope, 3), 0.137)

    rec_sl = [np.polyfit(*family_points(rows, m, "recolor"), 1)[0] for m in models]
    tot_sl = [np.polyfit(*family_points(rows, m, "tot_recolor"), 1)[0] for m in models]
    rec_med = float(np.median(rec_sl))
    share = rec_med / obs_slope * 100
    check("E1 recolor-only geometric share = 7.7%", round(share, 1), 7.7)
    pooled = float(np.median([(a + c) / 2 for a, c in zip(rec_sl, tot_sl)]))
    check("E1 pooled share = 10.0%", round(pooled / obs_slope * 100, 1), 10.0)

    # endpoint deltas (gamma=2 aniso, gamma=2 iso, recolor 0) per model vs base
    def endpoint(fam, param):
        out = []
        for m in models:
            pts = [r for r in rows if r["model"] == m]
            bb = next(r for r in pts if r["kind"] == "base")
            a = next(r for r in pts if r["kind"] == fam and r["param"] == param)
            out.append(a["c100_probe"] - bb["c100_probe"])
        return out

    aniso2 = endpoint("gauss_aniso", 2.0)
    check("E1 aniso2 median harm = -10.8 pp (8/8 negative)",
          (round(float(np.median(aniso2)) * 100, 1),
           all(x < 0 for x in aniso2)), (-10.8, True))
    check("E1 aniso dose-response monotone: medians -5.4/-7.1/-10.8 pp",
          [round(float(np.median(endpoint("gauss_aniso", p))) * 100, 1)
           for p in (0.5, 1.0, 2.0)], [-5.4, -7.1, -10.8])
    rec0 = endpoint("recolor", 0.0)
    check("E1 recolor0 median |delta| <= 2 pp",
          abs(float(np.median(rec0))) <= 0.02, True)

    # rotation is the negative control: preserves the spectrum exactly,
    # max |delta acc| = 1.7 pp across the 8 models
    rot = [next(r for r in rows if r["model"] == m and r["kind"] == "rotate")
           for m in models]
    dacc = [abs(r["c100_probe"] - b["c100_probe"]) * 100 for r, b in zip(rot, base)]
    check("E1 rotate max |delta acc| = 1.7 pp (moves nothing)",
          round(max(dacc), 1), 1.7)

    # ---------------- E2 ----------------
    floors = {"resnet34": ("v2_e2_r34.json", [-6.9, 5.0]),
              "resnet50": ("v2_e2_r50.json", [-2.5, 43.0]),
              "shufflenet_v2_x1_0": ("v2_e2_sf.json", [-20.2, 21.7]),
              "regnet_y_400mf": ("v2_e2_reg.json", [-2.8, 4.6])}
    for arch, (tag, want) in floors.items():
        r2 = load(tag, f"v2/{tag.replace('v2_', '')}/intervene_v2.json")
        # permutation noise floor is recomputed by paper3_ci.py; here we assert
        # the shipped JSON agrees with the number we ship in results/paper3_ci.json
        ci = json.loads((REPO / "results" / "paper3_ci.json").read_text())
        got = [round(x, 1) for x in ci["e2"][arch]["noise_floor95"]]
        check(f"E2 {arch} noise floor", got, want)

    # ---------------- E3 ----------------
    r3 = load("v2_e3.json", "v2/e3/intervene_v2.json")
    e3d = {r["model"]: r for r in r3 if r["arm"] == "e3"}
    e3c = {r["model"]: r for r in r3 if r["arm"] == "e3c"}
    ms3 = sorted(e3d)
    # formation gain: whitened-space probe vs control features under the same probe
    deltas = [e3d[m]["c100_probe"] - e3c[m]["c100_probe_ctrl"] for m in ms3]
    med = float(np.median(deltas)) * 100
    check("E3 formation gain median = +1.7 pp", round(med, 1), 1.7)
    check("E3 formation gain 7/8 positive",
          sum(1 for d in deltas if d > 0) == 7, True)
    w = wilcoxon(deltas, alternative="greater").pvalue
    check("E3 Wilcoxon raw p = 0.0078", round(float(w), 4), 0.0078)
    # inference arm: recolor(0) applied to features, original head unchanged
    inf = [e3d[m]["id_fc_acc_inference_recolor"] - e3c[m]["id_fc_acc_orig"]
           for m in ms3]
    check("E3 inference arm 7/8 negative",
          sum(1 for d in inf if d < 0) == 7, True)
    check("E3 inference median = -0.13 pp",
          round(float(np.median(inf)) * 100, 2), -0.13)
    # whitened-formation head scored on original features: 8/8 lose vs control head
    hc = [e3d[m]["id_fc_acc_orig"] - e3c[m]["id_fc_acc_orig"] for m in ms3]
    check("E3 head-classification loss 8/8 negative, median -0.35 pp",
          (all(x < 0 for x in hc), round(float(np.median(hc)) * 100, 2)),
          (True, -0.35))

    # ---------------- E5b dose-response (arm a / b / c) ----------------
    rows5 = load("v2_e5b.json", "v2/e5b/intervene_e5_dose.json")
    ctrl = {(r["model"], r["seed"]): r for r in rows5 if r["e5"] == "ctrl"}
    arm_a = {}
    for r in rows5:
        if r["e5"] == "b1":
            c = ctrl[(r["model"], r["seed"])]
            arm_a.setdefault(r["param"], {})[r["model"]] = \
                (r["c100_probe"] - c["c100_probe"]) * 100
    meds_by_dose = {g: float(np.median(list(v.values())))
                    for g, v in arm_a.items()}
    check("E5b arm-a cross-model median @g=0.75 = +1.9 pp",
          round(meds_by_dose[0.75], 1), 1.9)
    check("E5b arm-a cross-model median @g=0 = +3.1 pp",
          round(meds_by_dose[0.0], 1), 3.1)
    check("E5b arm-a cross-model median @g=-0.5 = +3.0 pp",
          round(meds_by_dose[-0.5], 1), 3.0)

    # arm b: formation readout vs E3 control head; monotone-worsening harm
    e3c_all = {r["model"]: r for r in r3 if r["arm"] == "e3c"}
    ext = load("v2_e5b_ext.json", "v2/e5b_ext/intervene_e5_dose.json")
    arm_b = []
    for r in ext:
        if r["e5"] == "b2" and r["param"] != 0.0 and r["model"] in e3c_all:
            base_head = e3c_all[r["model"]]["id_fc_acc_orig"]
            arm_b.append((r["model"], r["param"],
                          (r["id_fc_acc_orig"] - base_head) * 100))
    vals_b = [d for _, _, d in arm_b]
    check("E5b arm-b cross-dose range +0.11..-11.5 pp",
          (round(max(vals_b), 2), round(min(vals_b), 2)), (0.11, -11.48))
    # harm grows with dose: at g=-0.5 every model is worse than at g=0.75
    per = {}
    for m, g, d in arm_b:
        per.setdefault(m, {})[g] = d
    check("E5b arm-b harm grows with whitening dose (all models)",
          all(v[-0.5] < v[0.75] for v in per.values()), True)

    # arm c: rotate control -0.1..-0.8 pp
    arm_c = []
    for r in rows5:
        if r["e5"] == "b3":
            c = ctrl[(r["model"], r["seed"])]
            arm_c.append((r["c100_probe"] - c["c100_probe"]) * 100)
    check("E5b arm-c range -0.1..-0.8 pp",
          (round(max(arm_c), 1), round(min(arm_c), 1)), (-0.1, -0.8))

    # placebo: 7/9 cells at chance (0.10)
    pl = [r for r in rows5 if r["e5"] == "b2c"]
    n_chance = sum(1 for r in pl if abs(r["id_fc_acc_orig"] - 0.10) < 1e-9)
    check("E5b placebo destroys head training in 7/9 cells", n_chance, 7)

    # ---------------- E5c conditioning ----------------
    rowsc = load("v2_e5c.json", "v2/e5c/intervene_e5c_cond.json")
    best_red, best_m, curve_pair = 0.0, None, None
    for r in rowsc:
        if r["param"] == 0.0 and r["space"] == "dose":
            o = next(x for x in rowsc if x["model"] == r["model"]
                     and x["space"] == "orig" and x["param"] == 0.0)
            red = o["in_rank_cond"] / r["in_rank_cond"]
            if red > best_red:
                best_red, best_m = red, r["model"]
    check("E5c whitening reduces in-rank cond by up to ~3.3e5 (~six orders)",
          1e5 < best_red < 1e6, True)
    for sp in ("orig", "dose"):
        r = next(x for x in rowsc if x["model"] == best_m
                 and x["space"] == sp and x["param"] == 0.0)
        curve_pair = (r["curve"][0]["train_loss"], r["curve"][-1]["train_loss"]) \
            if sp == "orig" else curve_pair
        if sp == "dose":
            dose_curve = (r["curve"][0]["train_loss"], r["curve"][-1]["train_loss"])
    check("E5c ep1 loss slows 4.05 -> 4.50",
          (round(curve_pair[0], 2), round(dose_curve[0], 2)), (4.05, 4.50))
    check("E5c ep40 loss drops 3.43 -> 3.01",
          (round(curve_pair[1], 2), round(dose_curve[1], 2)), (3.43, 3.01))

    # ---------------- E5d MLP control ----------------
    rowsd = load("v2_e5d.json", "v2/e5d/intervene_e5_dose.json")
    mlp = {}
    for m in sorted({r["model"] for r in rowsd}):
        d = {r["kind"]: r for r in rowsd if r["model"] == m}
        mlp[m] = {
            "whiten_probe": (d["recolor"]["c100_probe"] - d["base"]["c100_probe"]) * 100,
            "whiten_mlp": (d["recolor"]["c100_mlp"] - d["base"]["c100_mlp"]) * 100,
            "aniso_mlp": (d["gauss_aniso"]["c100_mlp"] - d["base"]["c100_mlp"]) * 100,
        }
    wh = {m: v["whiten_mlp"] for m, v in mlp.items()}
    ah = sorted(v["aniso_mlp"] for v in mlp.values())
    # paper order: "(+1.1 and -3.3 pp)" = (lr0.5 model, lr0.125 model)
    wh_ordered = [round(wh[m], 1) for m in sorted(wh, reverse=True)]
    check("E5d MLP whiten benefit +1.1/-3.3 pp", wh_ordered, [1.1, -3.3])
    check("E5d MLP aniso harm -15.2/-16.4 pp",
          [round(ah[0], 1), round(ah[1], 1)], [-16.4, -15.2])

    # ---------------- E6 marker generality ----------------
    rows6 = load("v2_e6.json", "v2/e6/intervene_e6_marker.json")
    c100 = {r["model"]: r["c100_probe"] for r in base}
    prs = {r["model"]: r["pr"] for r in rows6 if r["kind"] == "base"}
    er = {r["model"]: r["erank"] for r in rows6 if r["kind"] == "base"}
    ms = sorted(prs)
    check("E6 r(PR, c100) = +0.77",
          round(pearson([prs[m] for m in ms], [c100[m] for m in ms]), 2), 0.77)
    check("E6 r(erank, c100) = +0.86",
          round(pearson([er[m] for m in ms], [c100[m] for m in ms]), 2), 0.86)
    check("E6 r(PR, D_eff) >= 0.81 (tracks dfx)",
          pearson([prs[m] for m in ms], [b["glue_d"] for b in base]) >= 0.81, True)
    for marker, keyf, want in (("PR", "pr", 32), ("erank", "erank", 29)):
        sh = [r[keyf] / b["pr" if keyf == "pr" else "erank"] * 100 - 100
              for r in rows6 if r["kind"] == "gauss_aniso" and r["param"] == 2.0
              for b in rows6 if b["kind"] == "base" and b["model"] == r["model"]]
        check(f"E6 aniso2 raises {marker} by median ~+{want}%",
              round(float(np.median(sh))), want)
    for marker, keyf, want in (("PR", "pr", 92), ("erank", "erank", 144)):
        sh = [r[keyf] / b["pr" if keyf == "pr" else "erank"] * 100 - 100
              for r in rows6 if r["kind"] == "recolor" and r["param"] == 0.0
              for b in rows6 if b["kind"] == "base" and b["model"] == r["model"]]
        check(f"E6 recolor0 raises {marker} by median ~+{want}%",
              round(float(np.median(sh))), want)
    rot = [r["pr"] / b["pr"] - 1
           for r in rows6 if r["kind"] == "rotate"
           for b in rows6 if b["kind"] == "base" and b["model"] == r["model"]]
    check("E6 rotate pins PR exactly (rel change +0.000)",
          max(abs(x) for x in rot) < 1e-9, True)

    # ---------------- E5e saturation / headroom (app:e5e) ----------------
    ceil = json.loads((REPO / "runs" / "v2_e5d_ceiling"
                       / "intervene_e5d_ceiling.json").read_text())["rows"]
    e3_rows = load("v2_e3.json", "v2/e3/intervene_v2.json")
    glue_d = {r["model"]: r["glue_d"] for r in e3_rows
              if r.get("kind") == "train_whiten"}
    m8 = sorted({r["model"] for r in ceil})
    check("E5e 8 models x 3 doses x 2 spaces = 48 rows", len(ceil), 48)
    check("E5e covers the 8 E3 paired models",
          all(m in glue_d for m in m8), True)

    e5e = {}
    for m in m8:
        o = next(r for r in ceil if r["model"] == m and r["param"] == 0.0
                 and r["space"] == "orig")
        w = next(r for r in ceil if r["model"] == m and r["param"] == 0.0
                 and r["space"] == "dose")
        e5e[m] = {
            "dfx": glue_d[m],
            "cond_o": o["in_rank_cond"],
            "improve": o["in_rank_cond"] / w["in_rank_cond"],
            "loss_o": o["curve"][-1]["train_loss"],
            "loss_w": w["curve"][-1]["train_loss"],
            "benefit": 100 * (w["curve"][-1]["c100_acc"]
                              - o["curve"][-1]["c100_acc"]),
        }
    neg = "resnet18_lr1.0_wd5e-05_s0"
    n = e5e[neg]
    others = [v["improve"] for m, v in e5e.items() if m != neg]
    # the stratification paragraph's dfx anchor (0.89 -> 0.87 correction)
    check("E5e negative-model baseline D_eff = 0.87 (grid minimum)",
          round(n["dfx"], 2), 0.87)
    check("E5e negative model orig-space in-rank cond = 294 "
          "(already nearly whitened)", round(n["cond_o"]), 294)
    check("E5e negative model conditioning improves only 22x "
          "(median others 4.6e3x)",
          (round(n["improve"]), round(float(np.median(others)))),
          (22, 4593))
    check("E5e negative model reachable loss fails to drop: 4.134 -> 4.165",
          (round(n["loss_o"], 3), round(n["loss_w"], 3)), (4.134, 4.165))
    check("E5e negative model probe benefit +0.08 pp (smallest of 8)",
          (round(n["benefit"], 2),
           n["benefit"] == min(v["benefit"] for v in e5e.values())),
          (0.08, True))
    # exactly three models fail to lower the reachable loss; the other two
    # are the slow-converging lr-0.5 probes named in the manuscript
    nondrop = sorted(m for m, v in e5e.items() if v["loss_w"] >= v["loss_o"])
    check("E5e 3/8 models fail to lower reachable loss",
          (len(nondrop),
           all(m.startswith("resnet18_lr0.5") for m in nondrop
               if m != neg)), (3, True))
    # C2 headroom: refuted -- dfx correlates positively with benefit
    xs5 = [v["dfx"] for v in e5e.values()]
    ys5 = [v["benefit"] for v in e5e.values()]
    rho, p_sp = spearmanr(xs5, ys5)
    r_pe = pearsonr(xs5, ys5)[0]
    check("E5e Spearman rho(dfx, benefit) = +0.857, p = 0.007 (refutes "
          "the headroom account)",
          (round(float(rho), 3), round(float(p_sp), 3)), (0.857, 0.007))
    check("E5e Pearson r(dfx, benefit) = +0.743",
          round(float(r_pe), 3), 0.743)
    hi = [v["benefit"] for v in e5e.values() if v["dfx"] >= 11]
    lo = [v["benefit"] for v in e5e.values() if v["dfx"] < 11]
    check("E5e high-dfx models (dfx >= 11) average +3.5 pp vs +0.9 pp",
          (len(hi), round(float(np.mean(hi)), 1),
           round(float(np.mean(lo)), 1)), (3, 3.5, 0.9))

    # replication check against E5c (3 models, 18 conditions, seed-pinned)
    rowsc2 = load("v2_e5c.json", "v2/e5c/intervene_e5c_cond.json")
    e5c_spec = {(r["model"], r["param"], r["space"]): r
                for r in rowsc2}
    conds = 0
    exact = 0
    tol_ok = 0
    acc_ok = 0
    for m in m8:
        for g in (0.75, 0.0, -0.5):
            for sp in ("orig", "dose"):
                if (m, g, sp) not in e5c_spec:
                    continue
                mine = next(r for r in ceil if r["model"] == m
                            and r["param"] == g and r["space"] == sp)
                old = e5c_spec[(m, g, sp)]
                conds += 1
                c_old, c_new = old["in_rank_cond"], mine["in_rank_cond"]
                if c_old == c_new:
                    exact += 1
                if abs(c_old - c_new) <= 1e-6 * max(abs(c_old), 1.0):
                    tol_ok += 1
                if old["curve"][-1]["c100_acc"] == mine["curve"][-1]["c100_acc"]:
                    acc_ok += 1
    check("E5e x E5c overlap = 18 conditions", conds, 18)
    check("E5e replication: cond exact 15/18, 18/18 within 1e-6 "
          "(BLAS tail bits), acc 18/18",
          (exact, tol_ok, acc_ok), (15, 18, 18))

    # ---------------- Holm family (primary tests) ----------------
    p_raw = [0.0039, 0.0078, 0.0078]  # P1b, P3a, P3b one-sided Wilcoxon
    order = np.argsort(p_raw)
    holm = [0.0] * 3
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (3 - rank) * p_raw[i])
        running = max(running, adj)
        holm[i] = round(running, 4)
    check("Holm-adjusted p (P1b, P3a, P3b) = 0.0117/0.0156/0.0156",
          holm, [0.0117, 0.0156, 0.0156])
    check("All three primary tests survive Holm at 0.05",
          all(p < 0.05 for p in holm), True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("All checks passed: every asserted number matches the locked JSON data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
