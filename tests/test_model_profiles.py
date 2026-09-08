"""CPU-only checks of the model-profile registry and per-record validation.

Builds, for every profile in the registry, a synthetic vector directory whose
arrays have that profile's shapes and whose digests are recomputed over the
actual bytes, and asserts that vectorfmt reads it back — and that it refuses a
record naming an unknown checkpoint, a record whose shape facts disagree with
its own checkpoint's profile, and an array whose shape belongs to a different
checkpoint than the record names.

Needs numpy and pytest, nothing else; no GPU, no vLLM.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from steering_vectors import vectorfmt
from steering_vectors.core import digest, modelprofile

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_META = REPO_ROOT / "vectors" / "0007" / "meta.json"


def synthetic_vector_dir(root: Path, profile: modelprofile.ModelProfile) -> Path:
    """One valid vector directory with ``profile``'s shapes, from the 0007 meta.

    The template supplies every field that is not a model fact (prompt format,
    capture record, conventions, description); the model facts, the arrays and
    the digests are rebuilt here so the directory is internally consistent.
    """
    meta = json.loads(TEMPLATE_META.read_text())
    layer = profile.n_layers // 2
    rng = np.random.default_rng(profile.hidden_size + profile.n_layers)
    deltas = rng.standard_normal(profile.deltas_shape).astype(np.float32)
    vector = deltas[layer].copy()

    vdir = root / meta["id_str"]
    vdir.mkdir(parents=True)
    with open(vdir / vectorfmt.VECTOR_NAME, "wb") as handle:
        np.save(handle, vector, allow_pickle=False)
    with open(vdir / vectorfmt.DELTAS_NAME, "wb") as handle:
        np.save(handle, deltas, allow_pickle=False)

    vector_norm = float(np.linalg.norm(vector.astype(np.float64)))
    activation_norm = 82.0
    meta.update(
        model=profile.model_id,
        n_layers=profile.n_layers,
        hidden_size=profile.hidden_size,
        layer=layer,
        vector_norm=vector_norm,
        activation_norm_at_layer=activation_norm,
        vector_norm_over_activation_norm=vector_norm / activation_norm,
        per_layer_delta_norm=[
            float(np.linalg.norm(row.astype(np.float64))) for row in deltas
        ],
        per_layer_mean_activation_norm=[activation_norm] * profile.n_layers,
        vector_npy_sha256=digest.sha256_file(vdir / vectorfmt.VECTOR_NAME),
        deltas_npy_sha256=digest.sha256_file(vdir / vectorfmt.DELTAS_NAME),
    )
    (vdir / vectorfmt.META_NAME).write_text(json.dumps(meta))
    return vdir


@pytest.mark.parametrize(
    "profile", modelprofile.PROFILES.values(), ids=sorted(modelprofile.PROFILES)
)
def test_every_profile_round_trips(tmp_path: Path, profile) -> None:
    vdir = synthetic_vector_dir(tmp_path, profile)
    meta = vectorfmt.read_meta(vdir)
    assert meta["model"] == profile.model_id
    vector = vectorfmt.load_vector(vdir, meta)
    deltas = vectorfmt.load_deltas(vdir, meta)
    assert vector.shape == profile.vector_shape
    assert deltas.shape == profile.deltas_shape
    vectorfmt.check_vector_is_delta_row(vector, deltas, meta["layer"])


def test_unknown_model_is_refused(tmp_path: Path) -> None:
    profile = modelprofile.QWEN3_5_9B
    vdir = synthetic_vector_dir(tmp_path, profile)
    meta = json.loads((vdir / vectorfmt.META_NAME).read_text())
    meta["model"] = "Qwen/Qwen-Imaginary-1B"
    (vdir / vectorfmt.META_NAME).write_text(json.dumps(meta))
    with pytest.raises(vectorfmt.VectorFormatError, match="has no profile"):
        vectorfmt.read_meta(vdir)


def test_shape_facts_must_match_the_recorded_models_profile(tmp_path: Path) -> None:
    # A 9B record wearing 2B shape facts: the record's own checkpoint decides.
    profile = modelprofile.QWEN3_5_2B
    vdir = synthetic_vector_dir(tmp_path, profile)
    meta = json.loads((vdir / vectorfmt.META_NAME).read_text())
    meta["model"] = modelprofile.QWEN3_5_9B.model_id
    (vdir / vectorfmt.META_NAME).write_text(json.dumps(meta))
    with pytest.raises(vectorfmt.VectorFormatError, match="disagree with"):
        vectorfmt.read_meta(vdir)


def test_array_of_another_profiles_shape_is_refused(tmp_path: Path) -> None:
    # Internally consistent 9B metadata over arrays with 2B shapes. The digests
    # match the bytes on disk, so it is the shape check that must refuse.
    vdir = synthetic_vector_dir(tmp_path, modelprofile.QWEN3_5_2B)
    wrong = modelprofile.QWEN3_5_9B
    meta = json.loads((vdir / vectorfmt.META_NAME).read_text())
    meta["model"] = wrong.model_id
    meta["n_layers"] = wrong.n_layers
    meta["hidden_size"] = wrong.hidden_size
    meta["layer"] = wrong.n_layers // 2
    pad = wrong.n_layers - len(meta["per_layer_delta_norm"])
    meta["per_layer_delta_norm"] += [1.0] * pad
    meta["per_layer_delta_norm"][meta["layer"]] = meta["vector_norm"]
    meta["per_layer_mean_activation_norm"] += [82.0] * pad
    (vdir / vectorfmt.META_NAME).write_text(json.dumps(meta))
    record = vectorfmt.read_meta(vdir)  # metadata alone is consistent
    with pytest.raises(vectorfmt.VectorFormatError, match="has shape"):
        vectorfmt.load_vector(vdir, record)
    with pytest.raises(vectorfmt.VectorFormatError, match="has shape"):
        vectorfmt.load_deltas(vdir, record)


def test_committed_vector_0007_still_reads() -> None:
    meta = vectorfmt.read_meta(REPO_ROOT / "vectors" / "0007")
    assert meta["model"] == "Qwen/Qwen3.6-27B"


def test_builder_refuses_a_model_other_than_the_selected_profile() -> None:
    import argparse

    from steering_vectors.build import cli

    args = argparse.Namespace(
        layer=1,
        model=modelprofile.QWEN3_5_9B.model_id
        if cli.PROFILE.model_id != modelprofile.QWEN3_5_9B.model_id
        else modelprofile.QWEN3_6_27B.model_id,
        positive="does-not-exist.jsonl",
        negative="does-not-exist.jsonl",
        name="x",
        description="x",
        vector_id=None,
        max_model_len=4096,
        vectors_dir=None,
    )
    with pytest.raises(ValueError, match="profiled only for"):
        cli.build(args)


def _selected_model_id(env_value: str | None) -> str:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    env.pop(modelprofile.PROFILE_ENV, None)
    if env_value is not None:
        env[modelprofile.PROFILE_ENV] = env_value
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from steering_vectors.core import modelprofile;"
            "print(modelprofile.PROFILE.model_id)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_selection_defaults_to_qwen3_6_27b() -> None:
    assert _selected_model_id(None) == "Qwen/Qwen3.6-27B"


@pytest.mark.parametrize("model_id", sorted(modelprofile.PROFILES))
def test_selection_by_env_var(model_id: str) -> None:
    assert _selected_model_id(model_id) == model_id
    short = model_id.removeprefix("Qwen/")
    assert _selected_model_id(short) == model_id


def test_selection_refuses_an_unknown_name() -> None:
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        _selected_model_id("Qwen/NoSuchModel")
    assert "names no known model profile" in excinfo.value.stderr
