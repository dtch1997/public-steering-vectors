"""Every model-specific fact the vector format checks against, in one object.

Nothing else in this package hard-codes a layer count, a hidden size or a
checkpoint name. Those facts are measurements taken against one checkpoint, and
scattering them as literals is what makes a model swap a search-and-replace
across the tree instead of one new :class:`ModelProfile`.

Swapping models is a matter of defining another profile in
:data:`PROFILES` and selecting it with the ``STEERING_MODEL_PROFILE``
environment variable (read once, at import) — after re-deriving the vectors,
because a vector is a difference of activations of one checkpoint and is not
comparable across two. Unset, the selection is :data:`QWEN3_6_27B`, so an
environment that predates the registry behaves exactly as it always did.

Deliberately contains only identity/shape facts plus the architecture facts the
capture-only builder needs. Serving and sampling facts remain outside this
profile.

Standard library only, and it holds no tensors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True, eq=False)
class ModelProfile:
    """The description of one checkpoint as this package uses it.

    Compared by identity: there is one instance, and two profiles that differ in
    any field describe different experiments.
    """

    # ---- identity -------------------------------------------------------
    model_id: str
    """The checkpoint the activations were captured from, as recorded in every
    vector's `meta.json` and as passed to the server that serves it."""

    dtype: str
    """Weight dtype the activations were measured at. Quantization is not an
    option here: it changes activations, and every norm, ratio and strength
    recorded with these vectors is a measurement of the unquantized residual
    stream."""

    # ---- shape ----------------------------------------------------------
    n_layers: int
    """Decoder blocks, indexed ``0 .. n_layers - 1``. Every layer argument in
    the package is bounds-checked against this."""

    hidden_size: int
    """Residual-stream width. A steering vector is ``(hidden_size,)`` float32
    and a full delta stack is ``(n_layers, hidden_size)``; a wrong-shaped array
    must be rejected when it is loaded rather than broadcast into a wrong
    answer."""

    # ---- capture architecture (verified against vLLM 0.26.0) ------------
    architecture: str
    """Checkpoint architecture key replaced by the capture plugin."""

    architecture_module: str
    """Private vLLM module containing the original architecture."""

    decoder_blocks_path: str
    """Attribute path from the architecture object to its decoder blocks."""

    residual_convention: str
    """How the true block-input residual stream is reconstructed."""

    verified_vllm_version: str
    """vLLM version against which the private replacement path was verified."""

    #: There is deliberately no default layer here. A layer is a property of the
    #: vector that was derived at it, recorded in that vector's metadata, and
    #: read from there by everything that steers or reads out with it. A
    #: model-level default would be a second, unowned source for the same
    #: number, and the failure it produces — steering at a layer the vector was
    #: not built for — is invisible in the output. Layer *bounds* are a model
    #: fact, and they are enforced by :meth:`check_layer`.

    notes: tuple[str, ...] = field(default=())
    """Facts with no natural field of their own."""

    # ---- derived shapes and validators ----------------------------------
    @property
    def vector_shape(self) -> tuple[int, ...]:
        """Shape of a single steering vector."""
        return (self.hidden_size,)

    @property
    def deltas_shape(self) -> tuple[int, ...]:
        """Shape of the per-layer delta stack produced by a derivation."""
        return (self.n_layers, self.hidden_size)

    def check_layer(self, layer: int, *, what: str = "layer") -> int:
        """Return ``layer`` if it indexes a decoder block, else raise.

        A layer index arrives from a command line and from a vector's metadata.
        Out of range, it either indexes from the end (negative) or fails deep
        inside the engine, where the message no longer mentions the layer.
        """
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise TypeError(
                f"{what} must be an int, got {type(layer).__name__}: {layer!r}"
            )
        if not 0 <= layer < self.n_layers:
            raise ValueError(
                f"{what} {layer} out of range 0..{self.n_layers - 1} "
                f"for {self.model_id} ({self.n_layers} decoder blocks)"
            )
        return layer

    def check_vector_shape(
        self, shape: tuple[int, ...], *, what: str = "vector"
    ) -> None:
        """Raise unless ``shape`` is this model's vector shape."""
        if tuple(shape) != self.vector_shape:
            raise ValueError(
                f"{what} has shape {tuple(shape)}, expected {self.vector_shape} "
                f"for {self.model_id} (hidden size {self.hidden_size})"
            )

    def check_deltas_shape(
        self, shape: tuple[int, ...], *, what: str = "deltas"
    ) -> None:
        """Raise unless ``shape`` is this model's per-layer delta shape."""
        if tuple(shape) != self.deltas_shape:
            raise ValueError(
                f"{what} has shape {tuple(shape)}, expected {self.deltas_shape} "
                f"for {self.model_id} ({self.n_layers} layers × {self.hidden_size})"
            )


QWEN3_6_27B = ModelProfile(
    model_id="Qwen/Qwen3.6-27B",
    dtype="bfloat16",
    n_layers=64,
    hidden_size=5120,
    # Capture facts ported from local v2-steering-tools, where they were verified
    # around commits ea65b2f, a486b53, d732a8c, 6508dab, and 6678408.
    architecture="Qwen3_5ForConditionalGeneration",
    architecture_module="vllm.model_executor.models.qwen3_5",
    decoder_blocks_path="language_model.model.layers",
    residual_convention=(
        "hidden_states if residual is None else hidden_states + residual"
    ),
    verified_vllm_version="0.26.0",
    notes=(
        "The residual stream is read and steered at the *input* of a block, so "
        "'layer L' throughout the vector format means 'before block L runs'. "
        "A server that hooks a block's *output* therefore steers layer L when "
        "it is pointed at block L-1; see vectorfmt.steer_layer.",
    ),
)

# ---------------------------------------------------------------------------
# the Qwen3.5 family (dense + MoE), for the model-scale generalization study
# ---------------------------------------------------------------------------
#
# Shape facts (n_layers, hidden_size, dtype) and the architecture name are read
# from each checkpoint's config.json on Hugging Face (2026-09-08). Capture
# facts: vLLM 0.26.0's registry maps BOTH "Qwen3_5ForConditionalGeneration" and
# "Qwen3_5MoeForConditionalGeneration" to the module "qwen3_5" (registry.py
# lines 573-576 at tag v0.26.0), and both classes there hold their decoder
# stack at `language_model.model.layers` (the MoE class's own MixtureOfExperts
# mixin iterates exactly that path). The residual convention is the one
# implemented by Qwen3_5DecoderLayer, shared by the dense and MoE text models
# in that module — the same block class the verified Qwen3.6-27B capture went
# through. None of the six has had an end-to-end capture verified yet; see the
# note attached to every profile.

_QWEN3_5_CAPTURE_NOTES = (
    "The residual stream is read and steered at the *input* of a block, so "
    "'layer L' throughout the vector format means 'before block L runs'. "
    "A server that hooks a block's *output* therefore steers layer L when "
    "it is pointed at block L-1; see vectorfmt.steer_layer.",
    "Capture facts (architecture module, blocks path, residual convention) "
    "were verified by reading vLLM v0.26.0 source, not yet by an end-to-end "
    "capture on this checkpoint; the first capture run's startup record is "
    "what confirms them (build.capture.check_capture_startup).",
)


def _qwen3_5(
    model_id: str, *, n_layers: int, hidden_size: int, moe: bool
) -> ModelProfile:
    """One Qwen3.5-family profile; the shared facts live in one place."""
    return ModelProfile(
        model_id=model_id,
        dtype="bfloat16",
        n_layers=n_layers,
        hidden_size=hidden_size,
        architecture=(
            "Qwen3_5MoeForConditionalGeneration"
            if moe
            else "Qwen3_5ForConditionalGeneration"
        ),
        # Same module for dense and MoE: vLLM 0.26.0 registers both
        # architectures out of vllm/model_executor/models/qwen3_5.py.
        architecture_module="vllm.model_executor.models.qwen3_5",
        decoder_blocks_path="language_model.model.layers",
        residual_convention=(
            "hidden_states if residual is None else hidden_states + residual"
        ),
        verified_vllm_version="0.26.0",
        notes=_QWEN3_5_CAPTURE_NOTES,
    )


QWEN3_5_2B = _qwen3_5("Qwen/Qwen3.5-2B", n_layers=24, hidden_size=2048, moe=False)
QWEN3_5_9B = _qwen3_5("Qwen/Qwen3.5-9B", n_layers=32, hidden_size=4096, moe=False)
QWEN3_5_27B = _qwen3_5("Qwen/Qwen3.5-27B", n_layers=64, hidden_size=5120, moe=False)
QWEN3_5_35B_A3B = _qwen3_5(
    "Qwen/Qwen3.5-35B-A3B", n_layers=40, hidden_size=2048, moe=True
)
QWEN3_5_122B_A10B = _qwen3_5(
    "Qwen/Qwen3.5-122B-A10B", n_layers=48, hidden_size=3072, moe=True
)
QWEN3_5_397B_A17B = _qwen3_5(
    "Qwen/Qwen3.5-397B-A17B", n_layers=60, hidden_size=4096, moe=True
)


# ---------------------------------------------------------------------------
# the registry and the selection
# ---------------------------------------------------------------------------

#: Every profile this package knows, keyed by checkpoint id. The key is the
#: string a vector's `meta.json` records under "model", which is what lets
#: vectorfmt validate a vector against the profile of the checkpoint it was
#: actually derived from rather than against whichever profile is selected.
PROFILES: dict[str, ModelProfile] = {
    profile.model_id: profile
    for profile in (
        QWEN3_6_27B,
        QWEN3_5_2B,
        QWEN3_5_9B,
        QWEN3_5_27B,
        QWEN3_5_35B_A3B,
        QWEN3_5_122B_A10B,
        QWEN3_5_397B_A17B,
    )
}

#: Selects :data:`PROFILE` at import. Accepts a checkpoint id, with or without
#: its "Qwen/" owner prefix. Unset or empty means :data:`QWEN3_6_27B`.
PROFILE_ENV = "STEERING_MODEL_PROFILE"


def profile_for(model_id: str) -> ModelProfile | None:
    """The profile for a checkpoint id, or ``None`` if none is defined.

    ``None`` rather than a raise, because the two callers want different
    errors: vectorfmt refuses the *vector* (its recorded checkpoint is one this
    package holds no facts about), while selection refuses the *environment*.
    """
    return PROFILES.get(model_id)


def _select_profile(
    env: os._Environ[str] | dict[str, str] = os.environ,
) -> ModelProfile:
    name = (env.get(PROFILE_ENV) or "").strip()
    if not name:
        return QWEN3_6_27B
    profile = PROFILES.get(name) or PROFILES.get(f"Qwen/{name}")
    if profile is None:
        raise ValueError(
            f"{PROFILE_ENV}={name!r} names no known model profile. "
            f"Known: {', '.join(sorted(PROFILES))}."
        )
    return profile


#: The profile every module uses: the *selected* checkpoint — the one being
#: captured from or served. One instance, named separately from the checkpoint
#: it describes so that call sites read as "the model", not "Qwen". Bound at
#: import from ``STEERING_MODEL_PROFILE``; vectors are validated against their
#: own recorded checkpoint's profile (:func:`profile_for`), not against this.
PROFILE = _select_profile()
