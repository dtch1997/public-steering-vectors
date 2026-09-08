"""Emit a random-control sibling of an existing vector directory.

A random control answers "is the effect the *direction*, or just the size of
the perturbation?": it is a seeded random Gaussian direction scaled to exactly
the source vector's norm, recorded against the same checkpoint, the same layer
and the same ``activation_norm_at_layer`` — so at any ``steer_strength`` it
perturbs the residual stream by the same relative magnitude as the real
vector, in a direction nothing was measured along.

    python scripts/make_random_control.py vectors/0007 --id 8 --seed 0

The output directory passes every ``vectorfmt`` read and integrity check and
is servable by the pod store alongside its source (same checkpoint, same
layer, hence the same hooked block). Facts that describe a derivation rather
than an array — the prompt files, ``prompt_format``, ``capture``, the
templated first prompt — are copied verbatim from the source, because the
format requires them; the description says so, and says they were NOT used.
The seed is recorded in the description and in ``command`` (``meta.json``'s
key set is closed, so there is deliberately no ``seed`` key).

``deltas_all_layers.npy`` is zeros except for the control itself at its layer
row, which is what makes ``vector.npy is row `layer` of the stack`` literally
true and is honest about the other rows: no other layer's difference was
measured, and a zero row's norm says exactly that.

Exit codes: 0 ok, 1 refused (bad source, existing id, unknown model), 2 usage.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steering_vectors import vectorfmt  # noqa: E402
from steering_vectors.build.cli import _write_index  # noqa: E402
from steering_vectors.core import (  # noqa: E402
    canonjson,
    clock,
    digest,
    modelprofile,
    provenance,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="make-random-control",
        description=(
            "Write a random-control vector directory: a seeded Gaussian "
            "direction at exactly the source vector's norm, same checkpoint, "
            "same layer, same strength unit."
        ),
    )
    command.add_argument("source", type=Path, metavar="VECTOR_DIR")
    command.add_argument("--id", dest="vector_id", type=int, required=True)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--name", default=None, help="default: random-control-of-<id>")
    command.add_argument(
        "--vectors-dir",
        type=Path,
        default=None,
        help="output root (default: the source directory's parent)",
    )
    command.add_argument(
        "--no-index",
        action="store_true",
        help="do not regenerate INDEX.md in the output root",
    )
    return command


def control_description(
    meta: dict, seed: int, actual_norm: float, cosine: float
) -> str:
    return (
        f"RANDOM CONTROL at matched norm for vector {meta['id_str']} "
        f"('{meta['name']}'), model {meta['model']}. The served array is a "
        f"random Gaussian direction (numpy default_rng seed {seed}, "
        f"{meta['hidden_size']} standard normal draws, normalized in float64) "
        f"scaled to the source vector's norm ({actual_norm:.6f} as stored in "
        f"float32, vs source {float(meta['vector_norm']):.6f}), recorded at "
        f"the source's layer {meta['layer']} with the source's "
        f"activation_norm_at_layer — so any steer_strength produces the same "
        f"relative perturbation magnitude as the source vector, in a random "
        f"direction (cosine to source: {cosine:+.6f}). No activations were "
        f"captured for this directory: positive.jsonl, negative.jsonl, "
        f"prompt_format, capture and templated_first_positive_prompt are "
        f"copied VERBATIM from vector {meta['id_str']} because the format "
        f"requires them; they describe the SOURCE's derivation and were NOT "
        f"used here. deltas_all_layers.npy is zeros except the control at row "
        f"{meta['layer']}, so the vector-is-its-layer-row identity holds and "
        f"no unmeasured difference is reported at any other layer."
    )


def build(args: argparse.Namespace) -> Path:
    import numpy as np

    source = args.source
    src_meta = vectorfmt.read_meta(source)
    vectorfmt.check_files_present(source)
    vectorfmt.verify_digests(source, src_meta)
    src_vector = vectorfmt.load_vector(source, src_meta)
    src_deltas = vectorfmt.load_deltas(source, src_meta)
    vectorfmt.check_vector_is_delta_row(src_vector, src_deltas, src_meta["layer"])
    profile = modelprofile.PROFILES[src_meta["model"]]  # read_meta guarantees it

    root = args.vectors_dir or Path(source).resolve().parent
    id_str = vectorfmt.vector_id(args.vector_id)
    destination = root / id_str
    if destination.exists():
        raise vectorfmt.VectorFormatError(
            f"{destination} already exists; vector ids are never overwritten"
        )

    # The direction: seeded, normalized and scaled in float64, stored float32.
    # The recorded norm is the norm of what is actually on disk, computed the
    # way load_vector computes it, so the metadata describes the stored bytes.
    rng = np.random.default_rng(args.seed)
    gaussian = rng.standard_normal(profile.hidden_size)
    vector = (
        gaussian / np.linalg.norm(gaussian) * float(src_meta["vector_norm"])
    ).astype(np.float32)
    actual_norm = float(np.linalg.norm(vector.astype(np.float64)))
    cosine = float(
        np.dot(vector.astype(np.float64), src_vector.astype(np.float64))
        / (actual_norm * float(np.linalg.norm(src_vector.astype(np.float64))))
    )
    deltas = np.zeros(profile.deltas_shape, dtype=np.float32)
    deltas[src_meta["layer"]] = vector
    per_layer_delta_norm = [0.0] * profile.n_layers
    per_layer_delta_norm[src_meta["layer"]] = actual_norm

    name = args.name or f"random-control-of-{src_meta['id_str']}"
    command = shlex.join(
        [
            "python",
            "scripts/make_random_control.py",
            str(source),
            "--id",
            str(args.vector_id),
            "--seed",
            str(args.seed),
        ]
    )

    destination.mkdir(parents=True)
    try:
        with open(destination / vectorfmt.VECTOR_NAME, "wb") as handle:
            np.save(handle, vector, allow_pickle=False)
        with open(destination / vectorfmt.DELTAS_NAME, "wb") as handle:
            np.save(handle, deltas, allow_pickle=False)
        for prompt_file in (vectorfmt.POSITIVE_NAME, vectorfmt.NEGATIVE_NAME):
            shutil.copyfile(source / prompt_file, destination / prompt_file)

        meta = dict(src_meta)
        act_norm = float(src_meta["activation_norm_at_layer"])
        meta.update(
            id=int(args.vector_id),
            id_str=id_str,
            name=name,
            description=control_description(src_meta, args.seed, actual_norm, cosine),
            vector_norm=actual_norm,
            vector_norm_over_activation_norm=actual_norm / act_norm,
            per_layer_delta_norm=per_layer_delta_norm,
            created_at=clock.utc_now_iso(),
            git_sha=provenance.repo_git_sha(),
            command=command,
            vector_npy_sha256=digest.sha256_file(destination / vectorfmt.VECTOR_NAME),
            deltas_npy_sha256=digest.sha256_file(destination / vectorfmt.DELTAS_NAME),
        )
        canonjson.write_json(destination / vectorfmt.META_NAME, meta)
        (destination / vectorfmt.README_NAME).write_text(
            f"# vector {id_str} — `{name}`\n\n{meta['description']}\n"
        )

        # Read the finished directory back through every check a server runs.
        written = vectorfmt.read_meta(destination)
        vectorfmt.check_files_present(destination)
        vectorfmt.verify_digests(destination, written)
        control = vectorfmt.load_vector(destination, written)
        vectorfmt.check_vector_is_delta_row(
            control, vectorfmt.load_deltas(destination, written), written["layer"]
        )
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    if not args.no_index:
        _write_index(root)
    print(
        f"{destination}\n"
        f"  ||control|| = {actual_norm:.6f} (source {src_meta['vector_norm']:.6f}), "
        f"cosine to source = {cosine:+.6f}, seed = {args.seed}"
    )
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        build(args)
    except (vectorfmt.VectorFormatError, OSError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
