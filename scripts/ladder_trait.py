"""Ladder driver for trait_openended: one steered condition per eval() call.

Env configuration (all required unless noted):
    L_MODEL   e.g. steered/Qwen/Qwen3.5-2B
    L_VECTOR  real vector id, e.g. 1002
    L_CTRL    matched-norm random control id, e.g. 9002
    L_LOGROOT e.g. logs/ladder/2b/trait
    STEERED_BASE_URL picks the server (provider default :8000)

Conditions: real vector at [-0.3, 0.0, +0.2, +0.3] plus control at +0.3,
matching the rest of the scoped ladder battery. Thinking OFF (ladder primary
protocol). EPOCHS=3 trims judge cost vs the task-default 5; same for every
rung, so cross-scale comparisons stay within-protocol.

Resume: a condition whose log dir already contains a completed trait_openended
log is skipped (coarse — one log per condition dir).
"""

from __future__ import annotations

import os
from pathlib import Path

from inspect_ai import eval
from inspect_ai.model import GenerateConfig, get_model
from trait_openended import trait_openended

MODEL = os.environ["L_MODEL"]
VECTOR = os.environ["L_VECTOR"]
CTRL = os.environ["L_CTRL"]
LOG_ROOT = os.environ["L_LOGROOT"]

CONDITIONS = [(VECTOR, -0.3), (VECTOR, 0.0), (VECTOR, 0.2), (VECTOR, 0.3), (CTRL, 0.3)]

QUESTIONS_PER_TRAIT = 20
MAX_TOKENS = 16384
EPOCHS = 3
JUDGE_SAMPLES = 1
JUDGE_TEMPERATURE = 0.0
MAX_CONNECTIONS = 10
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20


def main() -> None:
    for vec, strength in CONDITIONS:
        tag = f"{vec}_{strength:+.1f}".replace(".", "p")
        log_dir = str(Path(LOG_ROOT) / tag)
        if any(Path(log_dir).glob("*.eval")):
            print(f"[{tag}] existing log, skip")
            continue
        model = get_model(
            MODEL,
            config=GenerateConfig(
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                extra_body={"top_k": TOP_K},
                max_connections=MAX_CONNECTIONS,
            ),
            steer_vector=vec,
            steer_strength=strength,
            enable_thinking=False,
        )
        task = trait_openended(
            questions_per_trait=QUESTIONS_PER_TRAIT,
            judge_samples=JUDGE_SAMPLES,
            judge_temperature=JUDGE_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            epochs=EPOCHS,
        )
        logs = eval(
            task,
            model=model,
            epochs=EPOCHS,
            log_dir=log_dir,
            max_connections=MAX_CONNECTIONS,
            score=True,
            fail_on_error=False,
        )
        for log in logs:
            print(f"[{tag}] {log.status}")
            for score in log.results.scores if log.results else []:
                metrics = ", ".join(
                    f"{name}={metric.value}" for name, metric in score.metrics.items()
                )
                print(f"  {score.scorer}/{score.name}: {metrics}")


if __name__ == "__main__":
    main()
