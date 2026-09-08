"""scripts/make_random_control.py: matched-norm random controls, end to end.

Runs the script as a subprocess against a synthetic source directory, reads
the control back through every vectorfmt check, and confirms the pod store
would serve source and control side by side.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from test_model_profiles import REPO_ROOT, synthetic_vector_dir

from steering_vectors import vectorfmt
from steering_vectors.core import modelprofile

SCRIPT = REPO_ROOT / "scripts" / "make_random_control.py"

sys.path.insert(0, str(REPO_ROOT / "pod" / "src"))
from vllm_steering import store  # noqa: E402


def complete_source(
    root: Path, profile: modelprofile.ModelProfile = modelprofile.QWEN3_5_9B
) -> Path:
    """A synthetic source that also passes check_files_present/verify_digests.

    The template meta records the sha256 of vector 0007's prompt files, so
    copying those files makes the digests true.
    """
    vdir = synthetic_vector_dir(root, profile)
    for name in (vectorfmt.POSITIVE_NAME, vectorfmt.NEGATIVE_NAME):
        (vdir / name).write_bytes((REPO_ROOT / "vectors" / "0007" / name).read_bytes())
    (vdir / vectorfmt.README_NAME).write_text("# synthetic source\n")
    return vdir


def run_script(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv], capture_output=True, text=True
    )


def make_control(source: Path, vector_id: int = 8, seed: int = 3) -> Path:
    result = run_script(str(source), "--id", str(vector_id), "--seed", str(seed))
    assert result.returncode == 0, result.stderr
    return source.parent / f"{vector_id:04d}"


def test_control_round_trips_at_matched_norm(tmp_path: Path) -> None:
    source = complete_source(tmp_path)
    control_dir = make_control(source)

    meta = vectorfmt.read_meta(control_dir)
    vectorfmt.check_files_present(control_dir)
    vectorfmt.verify_digests(control_dir, meta)
    control = vectorfmt.load_vector(control_dir, meta)
    deltas = vectorfmt.load_deltas(control_dir, meta)
    vectorfmt.check_vector_is_delta_row(control, deltas, meta["layer"])

    src_meta = vectorfmt.read_meta(source)
    src_vector = vectorfmt.load_vector(source, src_meta)

    # Same checkpoint, same layer (hence the same hooked block), same
    # activation norm: the same strength is the same-sized intervention.
    assert meta["model"] == src_meta["model"]
    assert meta["layer"] == src_meta["layer"]
    assert meta["activation_norm_at_layer"] == src_meta["activation_norm_at_layer"]
    assert vectorfmt.steer_layer(meta) == vectorfmt.steer_layer(src_meta)

    # Matched norm, random direction.
    src_norm = float(np.linalg.norm(src_vector.astype(np.float64)))
    control_norm = float(np.linalg.norm(control.astype(np.float64)))
    assert abs(control_norm - src_norm) <= 1e-4 * src_norm
    assert abs(control_norm - float(meta["vector_norm"])) <= 1e-12 * control_norm
    cosine = float(
        np.dot(control.astype(np.float64), src_vector.astype(np.float64))
        / (control_norm * src_norm)
    )
    assert abs(cosine) < 0.1
    assert (
        abs(
            vectorfmt.relative_perturbation(1.0, meta)
            - vectorfmt.relative_perturbation(1.0, src_meta)
        )
        <= 1e-4
    )

    # The record says what it is.
    assert "RANDOM CONTROL" in meta["description"]
    assert "seed 3" in meta["description"]
    assert meta["name"] == "random-control-of-0007"


def test_control_is_deterministic_in_the_seed(tmp_path: Path) -> None:
    dirs = []
    for sub in ("a", "b"):
        root = tmp_path / sub
        root.mkdir()
        dirs.append(make_control(complete_source(root)))
    first, second = (vectorfmt.read_meta(vdir)["vector_npy_sha256"] for vdir in dirs)
    assert first == second


def test_pod_store_serves_control_beside_source(tmp_path: Path) -> None:
    source = complete_source(tmp_path)
    make_control(source)
    served = store.read(tmp_path)
    assert served.ids == ("0007", "0008")
    assert served.model == modelprofile.QWEN3_5_9B.model_id
    rows = store.matrix(served, modelprofile.QWEN3_5_9B.hidden_size)
    assert rows.shape == (3, modelprofile.QWEN3_5_9B.hidden_size)
    # Both rows serve at the activation norm: strength 1.0 is the same-sized
    # intervention for the real vector and its control.
    act = float(vectorfmt.read_meta(source)["activation_norm_at_layer"])
    for row in (1, 2):
        assert abs(float(np.linalg.norm(rows[row].astype(np.float64))) - act) < 1e-3


def test_existing_id_is_refused(tmp_path: Path) -> None:
    source = complete_source(tmp_path)
    make_control(source)
    result = run_script(str(source), "--id", "8", "--seed", "3")
    assert result.returncode == 1
    assert "already exists" in result.stderr


def test_incomplete_source_is_refused(tmp_path: Path) -> None:
    vdir = synthetic_vector_dir(tmp_path, modelprofile.QWEN3_5_2B)  # no jsonl
    result = run_script(str(vdir), "--id", "9")
    assert result.returncode == 1
    assert "missing" in result.stderr
