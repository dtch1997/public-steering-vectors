"""Collate every steering-autograder-scale eval log into one ladder summary.

    python scripts/collate_ladder.py

Walks ALL .eval logs under logs/ (anchor sweeps, the scale ladder, and any
future rung directories that appear — discovery is generic, driven by what each
log says about itself rather than by a directory list) and writes:

  results/ladder_summary.jsonl  — one row per (model, eval, protocol, vector,
                                  strength) condition
  results/ladder_summary.md     — per-eval condition tables plus a
                                  real-vs-control effect table at +0.3 / −0.3

Re-runnable: point it at the same tree after new logs land (e.g. 122b / 397b
rungs with vectors 1122/9122, 1397/9397) and both outputs are rebuilt from
scratch.

Grouping is by what the log header records, not where the file sits: the model
name, the task, the steering condition (`model_args.steer_vector` /
`steer_strength`), and the protocol. Protocol is `nothink` when the log's path
contains "nothink" or the model was built with `enable_thinking=False` (the
trait battery records it; the TQA/SoRH nothink runs only signal it by
directory name), otherwise `thinking-on`. Agentic-misalignment logs split into
one eval per scenario (am_blackmail / am_leaking / am_murder), and samples pool
across block/ctrl subdirectories exactly as
`scripts/analyze_agentic_misalignment.py` pools a directory — e.g. the anchor's
block01-e10 + block02-e40 give n=50 per (scenario, strength).

Metric extraction mirrors the repo's per-eval analysis scripts:

  truthfulqa  — the `choice` scorer's accuracy, read from the log header
                (weighted-pooled if a condition somehow has several logs).
  am_*        — mean of the `harmfulness_scorer` per-sample `harmful` value
                (and `classifier_verdict`), computed over samples, matching
                analyze_agentic_misalignment.py.
  school_of_reward_hacks — per-sample metric/quality/gap means with truncated
                samples (the shared `school_of_reward_hacks.truncation.
                no_answer` predicate) EXCLUDED, matching
                analyze_school_of_reward_hacks.py; validity rates are over all
                samples.
  trait_openended — the `trait_expression` scorer's mean from the header, plus
                the per-trait breakdown (`trait_expression_by_trait_*`) into
                `metrics` (agreeableness / machiavellianism / psychopathy and
                the rest).

Only logs whose status is `success` contribute, matching the analysis scripts;
skipped logs are printed. The 2b ladder's thinking-on TQA run is a FAILED
protocol (the model never emits a parseable choice); its rows are kept, with
protocol "thinking-on" and a `note` saying the accuracy is invalid.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "inspect_evals"))

from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402
from school_of_reward_hacks.truncation import no_answer  # noqa: E402

LOG_ROOT = REPO / "logs"
OUT_DIR = REPO / "results"

# Reading full samples would pull every message and event; nothing here needs
# them (same exclusion the analysis scripts use).
EXCLUDE_FIELDS = {"messages", "events", "events_data", "store", "attachments"}

AM_SCORER = "harmfulness_scorer"
SORH_SCORER = "reward_hack_gap_scorer"

# Params per model size (billions). Total -> (total, active); MoE rungs have
# fewer active params than total.
PARAMS = {2: (2, 2), 9: (9, 9), 27: (27, 27), 122: (122, 10), 397: (397, 17)}

# The headline metric per eval, used by the .md tables and effect columns.
HEADLINE = {
    "truthfulqa": "accuracy",
    "am_blackmail": "harmful",
    "am_leaking": "harmful",
    "am_murder": "harmful",
    "school_of_reward_hacks": "gap",
    "trait_openended": "trait_expression",
}

EVAL_ORDER = [
    "truthfulqa",
    "am_blackmail",
    "am_leaking",
    "am_murder",
    "school_of_reward_hacks",
    "trait_openended",
]

LADDER_TQA_THINKING_NOTE = (
    "thinking-on TQA protocol FAILED on ladder rungs; accuracy invalid"
)

KEY_TRAITS = ["agreeableness", "machiavellianism", "psychopathy"]


# --------------------------------------------------------------------------- #
# model identity


def model_identity(model: str) -> tuple[str, float, float, str]:
    """(short name, params_total_b, params_active_b, generation) from a model id.

    e.g. "steered/Qwen/Qwen3.6-27B" -> ("Qwen3.6-27B", 27, 27, "3.6").
    Active params come from an explicit "-A<n>B" suffix when present, else from
    the PARAMS table (122B->10 active, 397B->17 active), else equal total.
    """
    name = model.split("/")[-1]
    gen_match = re.search(r"Qwen(\d+\.\d+)", name)
    generation = gen_match.group(1) if gen_match else ""
    size_match = re.search(r"-(\d+)B", name)
    total = float(size_match.group(1)) if size_match else float("nan")
    active_match = re.search(r"-A(\d+(?:\.\d+)?)B", name)
    if active_match:
        active = float(active_match.group(1))
    else:
        active = float(PARAMS.get(int(total), (total, total))[1]) if size_match else total
    return name, total, active, generation


# --------------------------------------------------------------------------- #
# statistics


def mean_stderr(values: list[float]) -> tuple[float | None, float | None]:
    """Sample mean and standard error of the mean; stderr needs n >= 2."""
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n < 2:
        return mean, None
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance / n)


def finite(value: Any) -> float | None:
    """A float, or None where the value is missing or the NaN unscored sentinel."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def pooled(pairs: list[tuple[float, int]]) -> float | None:
    """n-weighted mean over (value, n) pairs."""
    total = sum(n for _, n in pairs)
    if total == 0:
        return None
    return sum(v * n for v, n in pairs) / total


# --------------------------------------------------------------------------- #
# per-log extraction


@dataclass
class Group:
    """All the success logs of one (model, eval, protocol, vector, strength)."""

    model: str
    eval: str
    protocol: str
    vector: str
    strength: float
    logs: list[str] = field(default_factory=list)  # log file paths


def location_path(location: str) -> Path:
    """A log location (file:// URI or plain path) as a local Path."""
    if location.startswith("file://"):
        return Path(location[len("file://") :])
    return Path(location)


def protocol_for(location: str, model_args: dict[str, Any]) -> str:
    if "nothink" in location:
        return "nothink"
    if model_args.get("enable_thinking") is False:
        return "nothink"
    return "thinking-on"


def eval_name(task: str, task_args: dict[str, Any]) -> str:
    if task == "agentic_misalignment":
        return f"am_{task_args.get('scenario', 'unknown')}"
    return task


def discover_groups() -> dict[tuple[str, str, str, str, float], Group]:
    """Walk logs/ and bucket every success log into its condition group."""
    groups: dict[tuple[str, str, str, str, float], Group] = {}
    infos = sorted(list_eval_logs(str(LOG_ROOT), recursive=True), key=lambda i: i.name)
    if not infos:
        raise SystemExit(f"no eval logs found under {LOG_ROOT}")
    for info in infos:
        header = read_eval_log(info.name, header_only=True)
        rel = location_path(header.location).relative_to(REPO)
        if header.status != "success":
            print(f"skip ({header.status}): {rel}")
            continue
        model_args = header.eval.model_args or {}
        task_args = header.eval.task_args or {}
        vector = str(model_args.get("steer_vector"))
        strength = round(float(model_args.get("steer_strength", 0.0)), 4)
        key = (
            header.eval.model,
            eval_name(header.eval.task, task_args),
            protocol_for(str(rel), model_args),
            vector,
            strength,
        )
        groups.setdefault(key, Group(*key)).logs.append(str(rel))
    return groups


def header_scorer_metrics(log_path: str) -> dict[str, dict[str, float]]:
    """{scorer name: {metric: value}} from a log header, merged across entries.

    Inspect splits a dict-valued score into several results entries that share
    the scorer's reported name (e.g. trait_expression appears once with
    mean/stderr and once with the by-trait breakdown); merging keeps both.
    """
    header = read_eval_log(log_path, header_only=True)
    out: dict[str, dict[str, float]] = defaultdict(dict)
    completed = header.results.completed_samples if header.results else 0
    out["_"]["completed"] = float(completed)
    for score in header.results.scores if header.results else []:
        out[score.name].update({k: m.value for k, m in score.metrics.items()})
    return out


def collate_truthfulqa(group: Group) -> tuple[int, dict[str, float | None]]:
    parts: list[tuple[float, float, int]] = []  # accuracy, stderr, n
    for log_path in group.logs:
        scorers = header_scorer_metrics(str(REPO / log_path))
        n = int(scorers["_"]["completed"])
        choice = scorers.get("choice", {})
        parts.append((float(choice.get("accuracy", float("nan"))), float(choice.get("stderr", float("nan"))), n))
    n_total = sum(n for _, _, n in parts)
    acc = pooled([(a, n) for a, _, n in parts])
    if len(parts) == 1:
        stderr: float | None = parts[0][1]
    elif acc is not None and n_total > 0:
        stderr = math.sqrt(acc * (1 - acc) / n_total)  # pooled binary accuracy
    else:
        stderr = None
    return n_total, {"accuracy": acc, "accuracy_stderr": stderr}


def collate_trait(group: Group) -> tuple[int, dict[str, float | None]]:
    n_total = 0
    expr: list[tuple[float, int]] = []
    stderrs: list[tuple[float, int]] = []
    by_trait: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for log_path in group.logs:
        scorers = header_scorer_metrics(str(REPO / log_path))
        n = int(scorers["_"]["completed"])
        n_total += n
        te = scorers.get("trait_expression", {})
        if "mean" in te:
            expr.append((float(te["mean"]), n))
        if "stderr" in te:
            stderrs.append((float(te["stderr"]), n))
        for key, value in te.items():
            prefix = "trait_expression_by_trait_"
            if key.startswith(prefix):
                by_trait[key[len(prefix) :]].append((float(value), n))
    metrics: dict[str, float | None] = {
        "trait_expression": pooled(expr),
        # single-log conditions keep the header stderr; pooled conditions get
        # the conservative n-weighted quadrature combination
        "trait_expression_stderr": (
            stderrs[0][0]
            if len(stderrs) == 1
            else (
                math.sqrt(sum((s * n) ** 2 for s, n in stderrs)) / sum(n for _, n in stderrs)
                if stderrs
                else None
            )
        ),
    }
    for trait in sorted(by_trait):
        metrics[f"trait_{trait}"] = pooled(by_trait[trait])
    return n_total, metrics


def collate_am(group: Group) -> tuple[int, dict[str, float | None]]:
    """Mean `harmful` / `classifier_verdict` over pooled samples.

    Mirrors analyze_agentic_misalignment.py: per-sample values from the
    `harmfulness_scorer` dict score, unscored samples dropped from each mean.
    """
    harmful: list[float] = []
    verdict: list[float] = []
    n_samples = 0
    n_error = 0
    for log_path in group.logs:
        log = read_eval_log(str(REPO / log_path), exclude_fields=EXCLUDE_FIELDS)
        for sample in log.samples or []:
            n_samples += 1
            if sample.error is not None:
                n_error += 1
            score = (sample.scores or {}).get(AM_SCORER)
            value = score.value if score is not None else None
            if isinstance(value, dict):
                h = finite(value.get("harmful"))
                v = finite(value.get("classifier_verdict"))
                if h is not None:
                    harmful.append(h)
                if v is not None:
                    verdict.append(v)
    h_mean, h_stderr = mean_stderr(harmful)
    v_mean, v_stderr = mean_stderr(verdict)
    return n_samples, {
        "harmful": h_mean,
        "harmful_stderr": h_stderr,
        "classifier_verdict": v_mean,
        "classifier_verdict_stderr": v_stderr,
        "n_error": float(n_error),
    }


def collate_sorh(group: Group) -> tuple[int, dict[str, float | None]]:
    """metric/quality/gap means with truncated samples excluded.

    Mirrors analyze_school_of_reward_hacks.py: the shared `no_answer` predicate
    (recomputed from the output, so pre- and post-scorer-change logs read the
    same) excludes a sample from every score mean; the truncation rate itself
    is over all samples. NaN scores (judge failed to parse) drop out of the
    score means but count into `unparsed_rate`.
    """
    n_samples = 0
    n_truncated = 0
    scores: dict[str, list[float]] = {"metric": [], "quality": [], "gap": []}
    unparsed: list[float] = []
    for log_path in group.logs:
        log = read_eval_log(str(REPO / log_path), exclude_fields=EXCLUDE_FIELDS)
        for sample in log.samples or []:
            n_samples += 1
            if no_answer(sample.output):
                n_truncated += 1
                continue
            score = (sample.scores or {}).get(SORH_SCORER)
            value = score.value if score is not None else None
            if not isinstance(value, dict):
                continue
            for key in scores:
                v = finite(value.get(key))
                if v is not None:
                    scores[key].append(v)
            u = finite(value.get("unparsed"))
            if u is not None:
                unparsed.append(u)
    metrics: dict[str, float | None] = {}
    for key, values in scores.items():
        mean, stderr = mean_stderr(values)
        metrics[key] = mean
        metrics[f"{key}_stderr"] = stderr
    metrics["unparsed_rate"] = (sum(unparsed) / len(unparsed)) if unparsed else None
    metrics["truncated_rate"] = (n_truncated / n_samples) if n_samples else None
    metrics["n_scored"] = float(len(scores["gap"]))
    metrics["n_truncated"] = float(n_truncated)
    return n_samples, metrics


def collate_group(group: Group) -> tuple[int, dict[str, float | None]]:
    if group.eval == "truthfulqa":
        return collate_truthfulqa(group)
    if group.eval.startswith("am_"):
        return collate_am(group)
    if group.eval == "school_of_reward_hacks":
        return collate_sorh(group)
    if group.eval == "trait_openended":
        return collate_trait(group)
    raise ValueError(f"no collator for eval {group.eval!r}")


# --------------------------------------------------------------------------- #
# rows


def row_for(group: Group) -> dict[str, Any]:
    name, total, active, generation = model_identity(group.model)
    n, metrics = collate_group(group)
    log_dir = os.path.commonpath([str(Path(p).parent) for p in group.logs])
    row: dict[str, Any] = {
        "model": name,
        "params_total_b": total,
        "params_active_b": active,
        "generation": generation,
        "eval": group.eval,
        "protocol": group.protocol,
        "vector": group.vector,
        "is_control": group.vector.startswith("9"),
        "strength": group.strength,
        "n": n,
        "metrics": {
            k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()
        },
        "log_dir": log_dir,
    }
    if (
        group.eval == "truthfulqa"
        and group.protocol == "thinking-on"
        and log_dir.startswith("logs/ladder/")
    ):
        row["note"] = LADDER_TQA_THINKING_NOTE
    return row


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["params_total_b"],
        row["generation"],
        row["model"],
        EVAL_ORDER.index(row["eval"]) if row["eval"] in EVAL_ORDER else 99,
        row["protocol"],
        row["is_control"],
        row["vector"],
        row["strength"],
    )


# --------------------------------------------------------------------------- #
# markdown


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def strength_label(strength: float) -> str:
    return f"{strength:+.1f}"


def condition_table(rows: list[dict[str, Any]], metric: str) -> list[str]:
    """One table for one eval: rows (model, protocol, vector) x strength cols."""
    strengths = sorted({r["strength"] for r in rows})
    cells: dict[tuple[str, str, str], dict[float, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        cells[(r["model"], r["protocol"], r["vector"])][r["strength"]] = r
    header = (
        ["model", "protocol", "vector", "ctrl?"]
        + [strength_label(s) for s in strengths]
        + ["n/cell"]
    )
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for (model, protocol, vector), by_strength in sorted(
        cells.items(), key=lambda kv: (sort_key(next(iter(kv[1].values()))))
    ):
        ns = sorted({r["n"] for r in by_strength.values()})
        n_label = str(ns[0]) if len(ns) == 1 else f"{ns[0]}–{ns[-1]}"
        any_row = next(iter(by_strength.values()))
        vals = []
        for s in strengths:
            r = by_strength.get(s)
            vals.append(fmt(r["metrics"].get(metric)) if r else "")
        note = " †" if any(("note" in r) for r in by_strength.values()) else ""
        lines.append(
            "| "
            + " | ".join(
                [model + note, protocol, vector, "ctrl" if any_row["is_control"] else "real"]
                + vals
                + [n_label]
            )
            + " |"
        )
    return lines


def find(rows: list[dict[str, Any]], model: str, protocol: str, is_control: bool, strength: float) -> dict[str, Any] | None:
    matches = [
        r
        for r in rows
        if r["model"] == model
        and r["protocol"] == protocol
        and r["is_control"] == is_control
        and abs(r["strength"] - strength) < 1e-9
    ]
    return matches[0] if matches else None


def baseline_for(rows: list[dict[str, Any]], model: str, protocol: str) -> dict[str, Any] | None:
    """The strength-0.0 row (steering inert, so the vector doesn't matter);
    prefer the real vector's if both exist."""
    real = find(rows, model, protocol, False, 0.0)
    return real or find(rows, model, protocol, True, 0.0)


def effect_table(rows: list[dict[str, Any]], metric: str, strength: float) -> list[str]:
    """Real vs control effect at one strength: effect = metric − metric(0.0)."""
    combos = sorted(
        {(r["model"], r["protocol"]) for r in rows},
        key=lambda mp: sort_key(next(r for r in rows if (r["model"], r["protocol"]) == mp)),
    )
    header = [
        "model",
        "protocol",
        "baseline (0.0)",
        f"real {strength_label(strength)}",
        f"ctrl {strength_label(strength)}",
        "effect real",
        "effect ctrl",
        "real − ctrl",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    emitted = 0
    for model, protocol in combos:
        base = baseline_for(rows, model, protocol)
        real = find(rows, model, protocol, False, strength)
        ctrl = find(rows, model, protocol, True, strength)
        if base is None or (real is None and ctrl is None):
            continue
        b = base["metrics"].get(metric)
        rv = real["metrics"].get(metric) if real else None
        cv = ctrl["metrics"].get(metric) if ctrl else None
        er = (rv - b) if (rv is not None and b is not None) else None
        ec = (cv - b) if (cv is not None and b is not None) else None
        diff = (er - ec) if (er is not None and ec is not None) else None
        note = " †" if any(r and "note" in r for r in (base, real, ctrl)) else ""
        lines.append(
            "| "
            + " | ".join(
                [model + note, protocol, fmt(b), fmt(rv), fmt(cv), fmt(er, 3), fmt(ec, 3), fmt(diff, 3)]
            )
            + " |"
        )
        emitted += 1
    return lines if emitted else []


def trait_breakdown_table(rows: list[dict[str, Any]]) -> list[str]:
    header = ["model", "protocol", "vector", "ctrl?", "strength"] + KEY_TRAITS
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in sorted(rows, key=sort_key):
        lines.append(
            "| "
            + " | ".join(
                [
                    r["model"],
                    r["protocol"],
                    r["vector"],
                    "ctrl" if r["is_control"] else "real",
                    strength_label(r["strength"]),
                ]
                + [fmt(r["metrics"].get(f"trait_{t}"), 1) for t in KEY_TRAITS]
            )
            + " |"
        )
    return lines


EVAL_TITLES = {
    "truthfulqa": "TruthfulQA (mc1 accuracy)",
    "am_blackmail": "Agentic misalignment — blackmail (harmful rate)",
    "am_leaking": "Agentic misalignment — leaking (harmful rate)",
    "am_murder": "Agentic misalignment — murder (harmful rate)",
    "school_of_reward_hacks": "School of Reward Hacks (paired gap = metric − quality)",
    "trait_openended": "Trait open-ended (trait_expression, judge-scored 0–100)",
}


def render_markdown(rows: list[dict[str, Any]]) -> str:
    by_eval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_eval[r["eval"]].append(r)

    lines = [
        "# Steering-autograder scale ladder — summary",
        "",
        f"Collated from `logs/` by `scripts/collate_ladder.py`; {len(rows)} conditions.",
        "",
        "Effect tables: effect = metric(vector at ±0.3) − metric(baseline, strength 0.0);",
        "`real − ctrl` is the direction-specific real-vs-control difference.",
        "",
        f"† {LADDER_TQA_THINKING_NOTE}.",
        "",
    ]
    evals = [e for e in EVAL_ORDER if e in by_eval] + sorted(
        set(by_eval) - set(EVAL_ORDER)
    )
    for ev in evals:
        ev_rows = by_eval[ev]
        metric = HEADLINE.get(ev, "accuracy")
        lines += [f"## {EVAL_TITLES.get(ev, ev)}", ""]
        lines += [f"Headline metric: `{metric}` by steering strength.", ""]
        lines += condition_table(ev_rows, metric) + [""]
        for strength in (0.3, -0.3):
            table = effect_table(ev_rows, metric, strength)
            if table:
                lines += [f"### Real vs control at {strength_label(strength)}", ""]
                lines += table + [""]
        if ev == "trait_openended":
            lines += ["### Per-trait means (key traits)", ""]
            lines += trait_breakdown_table(ev_rows) + [""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #


def main() -> None:
    groups = discover_groups()
    print(f"{len(groups)} conditions discovered")
    rows = []
    for key in sorted(groups):
        group = groups[key]
        row = row_for(group)
        rows.append(row)
        print(
            f"  {row['model']:>14} {row['eval']:<22} {row['protocol']:<11} "
            f"{row['vector']} {strength_label(row['strength'])} n={row['n']}"
        )
    rows.sort(key=sort_key)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / "ladder_summary.jsonl"
    with jsonl_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    md_path = OUT_DIR / "ladder_summary.md"
    md_path.write_text(render_markdown(rows))
    print(f"\n{len(rows)} rows -> {jsonl_path}\nmarkdown -> {md_path}")


if __name__ == "__main__":
    main()
