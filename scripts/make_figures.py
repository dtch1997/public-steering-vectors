"""Figures for steering-autograder-scale, from results/ladder_summary.jsonl.

Run with the workspace root venv (has xy):
    /mnt/nw/home/d.tan/jarvis-monorepo/.venv/bin/python scripts/make_figures.py

Fig 1 — direction-specific emergence: per eval (small multiples), the change
vs baseline at +0.3 for the real grader vector next to the matched-norm
random control, per ladder rung (nothink protocol) + the Qwen3.6-27B anchor
(thinking-on for am/SoRH; nothink trait). Takeaway: real separates from
control as scale grows.

Fig 2 — murder harm vs steering strength, one line per model. Takeaway: the
dose-response curve steepens with scale.
"""

import json
from pathlib import Path

import xy.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ROWS = [json.loads(l) for l in open(ROOT / "results/ladder_summary.jsonl")]

# Okabe-Ito, fixed assignment per model (identity follows entity)
MODELS = ["Qwen3.5-2B", "Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-122B-A10B", "Qwen3.6-27B"]
LABELS = {"Qwen3.5-2B": "2B", "Qwen3.5-9B": "9B", "Qwen3.5-27B": "27B",
          "Qwen3.5-122B-A10B": "122B-A10B", "Qwen3.6-27B": "3.6-27B (anchor)"}
COLORS = {"Qwen3.5-2B": "#E69F00", "Qwen3.5-9B": "#56B4E9", "Qwen3.5-27B": "#009E73",
          "Qwen3.5-122B-A10B": "#CC79A7", "Qwen3.6-27B": "#000000"}

EVALS = [
    ("am_murder", "harmful", "Agentic misalignment (murder)\nΔ harmful rate"),
    ("school_of_reward_hacks", "gap", "School of Reward Hacks\nΔ gaming gap (metric − quality)"),
    ("truthfulqa", "accuracy", "TruthfulQA\nΔ accuracy"),
    ("trait_openended", "trait_expression", "Trait expression (open-ended)\nΔ mean expression"),
]


def get(model, eval_, metric, strength, control):
    """Metric value for one condition; prefers nothink, falls back thinking-on (anchor am/SoRH)."""
    cands = [r for r in ROWS if r["model"] == model and r["eval"] == eval_
             and abs(r["strength"] - strength) < 1e-9 and r["is_control"] == control
             and "invalid" not in str(r.get("note", ""))]
    if not cands:
        return None
    cands.sort(key=lambda r: (r["protocol"] != "nothink", -r["n"]))
    return cands[0]["metrics"].get(metric)


def fig1():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (ev, metric, title) in zip(axes.flat, EVALS):
        xs, reals, ctrls, names = [], [], [], []
        for i, m in enumerate(MODELS):
            base = get(m, ev, metric, 0.0, False)
            real = get(m, ev, metric, 0.3, False)
            ctrl = get(m, ev, metric, 0.3, True)
            if base is None or real is None:
                continue
            xs.append(len(xs))
            reals.append(real - base)
            ctrls.append((ctrl - base) if ctrl is not None else float("nan"))
            names.append(LABELS[m])
        w = 0.38
        ax.bar([x - w / 2 for x in xs], reals, width=w, color="#0072B2",
               label="grader vector (+0.3)")
        ax.bar([x + w / 2 for x in xs], ctrls, width=w, color="#BBBBBB",
               label="random control (+0.3)")
        ax.axhline(0, color="#888888", linewidth=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(title, fontsize=9)
        if ax is axes.flat[0]:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Automated-grader steering: direction-specific effects emerge with scale\n"
                 "(change vs unsteered baseline at strength +0.3; ladder = thinking-off, anchor am/SoRH = thinking-on)",
                 fontsize=10)
    fig.tight_layout()
    out = ROOT / "results/figures/fig1_specificity_by_scale.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(out)


def fig2():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    strengths = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    for m in MODELS:
        xs, ys = [], []
        for s in strengths:
            v = get(m, "am_murder", "harmful", s, False)
            if v is not None:
                xs.append(s)
                ys.append(v)
        if not xs:
            continue
        style = dict(color=COLORS[m], linewidth=2, marker="o", markersize=5)
        if m == "Qwen3.6-27B":
            style["linestyle"] = "--"
        ax.plot(xs, ys, label=LABELS[m], **style)
    ax.set_xlabel("steer strength (relative perturbation units)")
    ax.set_ylabel("murder scenario: harmful action rate")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Murder-scenario dose-response steepens with scale\n"
                 "(ladder rungs thinking-off; anchor dashed, thinking-on, n=50)", fontsize=10)
    fig.tight_layout()
    out = ROOT / "results/figures/fig2_murder_dose_response.png"
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    fig1()
    fig2()
