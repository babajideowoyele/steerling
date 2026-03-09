"""
4-bit quantization and hybrid offloading for Steerling models.

Three loading strategies for low-VRAM GPUs:

1. ``load_quantized``: NF4 via bitsandbytes, everything on GPU (~5.2 GB).
2. ``load_hybrid``: NF4 transformer on GPU + bf16 concept heads on CPU (~4.8 GB GPU).
3. ``load_torchao``: INT4 via torchao (PyTorch-native, no bitsandbytes needed).

Usage::

    from steerling.quantization import load_quantized, load_hybrid, load_torchao

    # bitsandbytes NF4 (pip install bitsandbytes)
    generator = load_quantized("guidelabs/steerling-8b")

    # Hybrid: NF4 on GPU + bf16 concept heads on CPU
    generator = load_hybrid("guidelabs/steerling-8b")

    # torchao INT4 — no bitsandbytes needed (pip install torchao)
    generator = load_torchao("guidelabs/steerling-8b")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from steerling.inference.causal_diffusion import SteerlingGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _replace_linear_with_4bit(
    module: nn.Module,
    *,
    skip_names: set[str] | None = None,
    current_name: str = "",
) -> None:
    """Recursively replace nn.Linear layers with bnb.nn.Linear4bit in-place.

    Args:
        module: The module to modify.
        skip_names: Dotted parameter names to skip (e.g. {"lm_head"}).
        current_name: Used internally for recursion tracking.
    """
    import bitsandbytes as bnb

    skip_names = skip_names or set()

    for name, child in list(module.named_children()):
        full_name = f"{current_name}.{name}" if current_name else name

        if full_name in skip_names:
            logger.debug(f"Skipping quantization of {full_name}")
            continue

        if isinstance(child, nn.Linear):
            has_bias = child.bias is not None
            quantized = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=has_bias,
                compute_dtype=torch.bfloat16,
                quant_type="nf4",
                compress_statistics=True,
            )
            quantized.weight = bnb.nn.Params4bit(
                child.weight.data,
                requires_grad=False,
                quant_type="nf4",
                compress_statistics=True,
            )
            if has_bias:
                quantized.bias = nn.Parameter(child.bias.data, requires_grad=False)

            setattr(module, name, quantized)
            logger.debug(f"Quantized {full_name}: {child.in_features}x{child.out_features}")
        else:
            _replace_linear_with_4bit(child, skip_names=skip_names, current_name=full_name)


def _count_params_by_dtype(model: nn.Module) -> dict[str, int]:
    """Count parameters grouped by dtype for logging."""
    counts: dict[str, int] = {}
    for p in model.parameters():
        key = str(p.dtype)
        counts[key] = counts.get(key, 0) + p.numel()
    return counts


def _assign_tensor(model: nn.Module, key: str, tensor: Tensor) -> None:
    """Assign a tensor to a model parameter by dotted key, materializing from meta."""
    parts = key.split(".")
    mod = model
    for part in parts[:-1]:
        mod = getattr(mod, part)
    param_name = parts[-1]
    old = getattr(mod, param_name)
    if isinstance(old, nn.Parameter):
        setattr(mod, param_name, nn.Parameter(tensor, requires_grad=False))
    else:
        setattr(mod, param_name, tensor)


def _resolve_weight_tying(
    base: nn.Module, state_dict: dict[str, Tensor], model_config: object
) -> None:
    """Handle weight tying after meta→CPU assignment."""
    if not getattr(model_config, "weight_sharing", False):
        return
    # If lm_head wasn't in state_dict, copy from tok_emb
    if hasattr(base, "lm_head") and base.lm_head.weight.device.type == "meta":
        if hasattr(base, "tok_emb") and base.tok_emb.weight.device.type != "meta":
            base.lm_head.weight = base.tok_emb.weight
    # If tok_emb wasn't in state_dict, copy from lm_head
    elif (
        hasattr(base, "tok_emb")
        and base.tok_emb.weight.device.type == "meta"
        and hasattr(base, "lm_head")
        and base.lm_head.weight.device.type != "meta"
    ):
        base.tok_emb.weight = base.lm_head.weight


def _estimate_vram_mb(model: nn.Module, device: str = "cuda") -> float:
    """Estimate VRAM usage from parameters on the given device."""
    total = 0
    target = torch.device(device)
    for p in model.parameters():
        if p.device.type == target.type:
            total += p.nelement() * p.element_size()
    return total / (1024 * 1024)


def _load_model_and_config(
    model_name_or_path: str,
) -> tuple[nn.Module, dict, bool]:
    """Shared model creation and weight loading (on CPU, bf16).

    Uses meta-device initialization + per-tensor weight assignment to
    keep peak RAM at ~17 GB instead of ~34 GB (model + state dict).

    Returns:
        (model, raw_config, is_interpretable)
    """
    import gc

    from steerling.configs.causal_diffusion import CausalDiffusionConfig
    from steerling.configs.concept import ConceptConfig
    from steerling.data.tokenizer import SteerlingTokenizer
    from steerling.inference.checkpoint_utils import load_config, load_state_dict
    from steerling.models.causal_diffusion import CausalDiffusionLM
    from steerling.models.interpretable.interpretable_causal_diffusion import (
        InterpretableCausalDiffusionLM,
    )

    raw_config = load_config(model_name_or_path)

    model_fields = set(CausalDiffusionConfig.model_fields.keys())
    model_data = {k: v for k, v in raw_config.items() if k in model_fields}
    model_config = CausalDiffusionConfig.model_validate(model_data)

    is_interpretable = raw_config.get("interpretable", False)
    concept_data = raw_config.get("concept")
    vocab_size = raw_config.get("vocab_size", SteerlingTokenizer().vocab_size)

    # Create model on meta device (no memory allocated for parameters)
    logger.info("Creating model on meta device...")
    with torch.device("meta"):
        if is_interpretable and concept_data is not None:
            concept_config = ConceptConfig.model_validate(concept_data)
            model: nn.Module = InterpretableCausalDiffusionLM(
                config=model_config,
                concept_config=concept_config,
                vocab_size=vocab_size,
            )
        else:
            model = CausalDiffusionLM(config=model_config, vocab_size=vocab_size)

    # Load state dict (only copy in RAM: ~17 GB)
    logger.info("Loading weights...")
    state_dict = load_state_dict(model_name_or_path)

    # Materialize model from meta → CPU by assigning state dict tensors
    logger.info("Assigning weights to model (meta → CPU)...")
    model_sd = model.state_dict()

    # Collect keys that exist in both model and checkpoint
    assigned = set()
    for key in list(model_sd.keys()):
        if key in state_dict:
            # Navigate to the parameter and replace it
            _assign_tensor(model, key, state_dict.pop(key).to(torch.bfloat16))
            assigned.add(key)

    # Handle weight tying: tok_emb.weight may need to come from lm_head or vice versa
    base = model.transformer if hasattr(model, "transformer") else model
    _resolve_weight_tying(base, state_dict, model_config)

    # Report missing / unexpected
    model_keys = set(model_sd.keys())
    missing = model_keys - assigned
    if missing:
        non_tying = [k for k in missing if "lm_head" not in k]
        if non_tying:
            logger.warning(f"Missing keys (non-tying): {non_tying}")
    unexpected = set(state_dict.keys())
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    # Free remaining state dict
    del state_dict, model_sd
    gc.collect()

    return model, raw_config, is_interpretable


def _break_weight_tying(model: nn.Module, model_config: object) -> None:
    """Break weight tying between tok_emb and lm_head for quantization."""
    base = model.transformer if hasattr(model, "transformer") else model
    if getattr(model_config, "weight_sharing", False):
        logger.info("Breaking weight tying for quantization...")
        base.lm_head.weight = nn.Parameter(base.tok_emb.weight.data.clone(), requires_grad=False)


def _finalize_generator(
    model: nn.Module,
    raw_config: dict,
    is_interpretable: bool,
    device: str,
    *,
    skip_device_move: bool = False,
) -> SteerlingGenerator:
    """Disable compiled flex_attention and wrap model in SteerlingGenerator.

    Args:
        skip_device_move: If True, prevent SteerlingGenerator.__init__ from
            calling model.to(device).  Used by load_hybrid where concept
            heads intentionally live on CPU.
    """
    from steerling.configs.causal_diffusion import CausalDiffusionConfig
    from steerling.data.tokenizer import SteerlingTokenizer
    from steerling.inference.causal_diffusion import SteerlingGenerator

    import steerling.models.layers.causal_diffusion_layers as layers
    from torch.nn.attention.flex_attention import flex_attention

    layers.compiled_flex_attention = flex_attention

    model_fields = set(CausalDiffusionConfig.model_fields.keys())
    model_data = {k: v for k, v in raw_config.items() if k in model_fields}
    model_config = CausalDiffusionConfig.model_validate(model_data)

    if skip_device_move:
        # Temporarily replace model.to so SteerlingGenerator.__init__
        # doesn't move concept heads off CPU.
        original_to = model.to
        model.to = lambda *args, **kwargs: model  # type: ignore[assignment]

    gen = SteerlingGenerator(
        model=model,
        tokenizer=SteerlingTokenizer(),
        model_config=model_config,
        is_interpretable=is_interpretable,
        device=device,
    )

    if skip_device_move:
        model.to = original_to  # type: ignore[assignment]

    return gen


# ---------------------------------------------------------------------------
# Strategy 1: Pure quantized (everything on GPU)
# ---------------------------------------------------------------------------


def load_quantized(
    model_name_or_path: str = "guidelabs/steerling-8b",
    device: str = "cuda",
) -> SteerlingGenerator:
    """Load a Steerling model with 4-bit NF4 quantization.

    Requires ``bitsandbytes`` (``pip install bitsandbytes``).
    Reduces VRAM from ~16.5 GB (bf16) to ~5.2 GB.

    Args:
        model_name_or_path: HuggingFace repo ID or local directory.
        device: Target CUDA device.

    Returns:
        SteerlingGenerator ready for inference.
    """
    try:
        import bitsandbytes as bnb  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "bitsandbytes is required for 4-bit quantization. Install it with: pip install bitsandbytes"
        ) from err

    from steerling.configs.causal_diffusion import CausalDiffusionConfig

    logger.info(f"Loading {model_name_or_path} with 4-bit quantization...")

    model, raw_config, is_interpretable = _load_model_and_config(model_name_or_path)

    model_fields = set(CausalDiffusionConfig.model_fields.keys())
    model_data = {k: v for k, v in raw_config.items() if k in model_fields}
    model_config = CausalDiffusionConfig.model_validate(model_data)

    _break_weight_tying(model, model_config)

    # Skip concept heads — their streaming code accesses .weight directly
    # for manual matmul, which is incompatible with bnb compressed format.
    _replace_linear_with_4bit(
        model,
        skip_names={"known_head", "unknown_head"},
    )

    logger.info(f"Moving to {device} (quantizing weights)...")
    model = model.to(device)
    model.eval()

    vram = _estimate_vram_mb(model, device)
    logger.info(f"Estimated VRAM: {vram:.0f} MB")

    return _finalize_generator(model, raw_config, is_interpretable, device)


# ---------------------------------------------------------------------------
# Strategy 2: Hybrid (transformer quantized on GPU, concept heads on CPU)
# ---------------------------------------------------------------------------


class _HybridInterpretableForward:
    """Replaces InterpretableCausalDiffusionLM.forward with device-aware version.

    Transformer + lm_head live on GPU. Concept heads live on CPU.
    Tensors are shuttled between devices at the boundary.
    """

    def __init__(self, original_model: nn.Module, gpu_device: torch.device):
        self.model = original_model
        self.gpu = gpu_device
        self.cpu = torch.device("cpu")

    def __call__(
        self,
        input_ids: Tensor,
        *,
        use_teacher_forcing: bool = False,
        intervene_known_ids: Tensor | None = None,
        intervene_known_vals: Tensor | None = None,
        intervene_unknown_ids: Tensor | None = None,
        intervene_unknown_vals: Tensor | None = None,
        minimal_output: bool = False,
        position_injection: Tensor | None = None,
        steering_inject_layer: int | None = None,
        steering_inject_alpha: float = 1.0,
        unknown_topk: int = 64,
    ) -> tuple[Tensor, object]:
        from steerling.models.interpretable.outputs import InterpretableOutput

        m = self.model
        need_dense_logits = not minimal_output

        # --- Transformer forward on GPU ---
        input_ids = input_ids.to(self.gpu)

        if position_injection is not None and steering_inject_layer is not None:
            hidden_gpu = m._forward_with_injection(
                input_ids,
                position_injection.to(self.gpu),
                steering_inject_layer,
                steering_inject_alpha,
            )
        else:
            hidden_gpu = m.transformer(input_ids, return_hidden=True)

        # --- Move hidden to CPU for concept heads ---
        hidden_cpu = hidden_gpu.to(self.cpu)

        # Move intervention tensors to CPU if present
        def _to_cpu(t: Tensor | None) -> Tensor | None:
            return t.to(self.cpu) if t is not None else None

        # --- Known concept head on CPU ---
        known_out = m.known_head(
            hidden_cpu,
            concept_ids=None,
            concept_mask=None,
            use_teacher_forcing=False,
            intervene_ids=_to_cpu(intervene_known_ids),
            intervene_vals=_to_cpu(intervene_known_vals),
            return_logits=need_dense_logits,
        )
        known_features_cpu = known_out.features.to(hidden_cpu.dtype)

        # Residual unknown
        unk_cpu = hidden_cpu - known_features_cpu.detach()

        # --- Unknown concept head on CPU ---
        unk_for_lm_cpu: Tensor = unk_cpu
        unknown_out = None
        unk_hat_cpu: Tensor | None = None

        if m.unknown_head is not None:
            unknown_out = m.unknown_head(
                hidden_cpu.detach(),
                intervene_ids=_to_cpu(intervene_unknown_ids),
                intervene_vals=_to_cpu(intervene_unknown_vals),
                return_logits=not minimal_output and not m.unknown_head._is_large,
            )
            unk_hat_cpu = unknown_out.features.to(hidden_cpu.dtype)
            unk_for_lm_cpu = unk_hat_cpu.detach()

        # Epsilon true
        epsilon_true = None
        if m.unknown_head is not None and unk_hat_cpu is not None:
            epsilon_true = hidden_cpu.detach() - (known_out.predicted + unk_hat_cpu)

        # Epsilon correction
        epsilon = None
        if m.concept_config.use_epsilon_correction and intervene_known_ids is None:
            epsilon = hidden_cpu - (unk_for_lm_cpu + known_features_cpu)
            unk_for_lm_cpu = unk_for_lm_cpu + epsilon

        # --- Compose on CPU, then project on GPU ---
        composed_cpu = unk_for_lm_cpu + known_features_cpu
        composed_gpu = composed_cpu.to(self.gpu)
        logits_gpu = m.transformer.lm_head(composed_gpu)

        # Unknown top-k attribution (on CPU)
        _unk_topk_indices = unknown_out.topk_indices if unknown_out else None
        _unk_topk_logits = unknown_out.topk_logits if unknown_out else None

        if (
            not minimal_output
            and m.unknown_head is not None
            and unknown_out is not None
            and _unk_topk_indices is None
            and unknown_topk > 0
        ):
            with torch.no_grad():
                _unk_topk_indices, _unk_topk_logits = m._compute_unknown_topk(hidden_cpu, unknown_topk)

        outputs = InterpretableOutput(
            hidden=hidden_gpu,
            known_features=known_features_cpu.to(self.gpu),
            known_logits=known_out.logits,
            known_gt_features=known_out.gt_features,
            known_predicted=known_out.predicted,
            known_weights=known_out.weights,
            known_topk_indices=known_out.topk_indices,
            known_topk_logits=known_out.topk_logits,
            unk=unk_cpu,
            unk_hat=unk_hat_cpu,
            unk_for_lm=unk_for_lm_cpu,
            unknown_logits=unknown_out.logits if unknown_out else None,
            unknown_weights=unknown_out.weights if unknown_out else None,
            unknown_topk_indices=_unk_topk_indices,
            unknown_topk_logits=_unk_topk_logits,
            composed=composed_gpu,
            epsilon=epsilon,
            epsilon_true=epsilon_true,
        )

        return logits_gpu, outputs


def load_hybrid(
    model_name_or_path: str = "guidelabs/steerling-8b",
    device: str = "cuda",
) -> SteerlingGenerator:
    """Load with 4-bit transformer on GPU + bf16 concept heads on CPU.

    Compared to ``load_quantized``:

    - ~400 MB less VRAM (~4.8 GB vs ~5.2 GB), leaving more room for
      activations and longer sequences.
    - Concept heads stay in bf16 — no quantization quality loss for
      attribution and steering.
    - Slightly slower per step due to GPU<->CPU data transfer for hidden
      states (~4 KB per token per step).

    Requires ``bitsandbytes`` (``pip install bitsandbytes``).

    Args:
        model_name_or_path: HuggingFace repo ID or local directory.
        device: Target CUDA device.

    Returns:
        SteerlingGenerator ready for inference.
    """
    try:
        import bitsandbytes as bnb  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "bitsandbytes is required for hybrid quantization. Install it with: pip install bitsandbytes"
        ) from err

    from steerling.configs.causal_diffusion import CausalDiffusionConfig
    from steerling.models.interpretable.interpretable_causal_diffusion import (
        InterpretableCausalDiffusionLM,
    )

    logger.info(f"Loading {model_name_or_path} with hybrid quantization...")

    model, raw_config, is_interpretable = _load_model_and_config(model_name_or_path)

    model_fields = set(CausalDiffusionConfig.model_fields.keys())
    model_data = {k: v for k, v in raw_config.items() if k in model_fields}
    model_config = CausalDiffusionConfig.model_validate(model_data)

    if not is_interpretable or not isinstance(model, InterpretableCausalDiffusionLM):
        logger.info("Model is not interpretable — falling back to load_quantized.")
        _break_weight_tying(model, model_config)
        _replace_linear_with_4bit(model)
        model = model.to(device)
        model.eval()
        return _finalize_generator(model, raw_config, is_interpretable, device)

    # --- Break weight tying ---
    _break_weight_tying(model, model_config)

    # --- Quantize ONLY the transformer (not concept heads) ---
    logger.info("Quantizing transformer blocks to NF4...")
    _replace_linear_with_4bit(
        model,
        skip_names={"known_head", "unknown_head"},
    )

    # --- Move transformer + lm_head + tok_emb to GPU ---
    logger.info(f"Moving transformer to {device}...")
    model.transformer.to(device)

    # --- Keep concept heads on CPU (bf16, no quantization) ---
    logger.info("Keeping concept heads on CPU (bf16)...")
    model.known_head.to("cpu")
    if model.unknown_head is not None:
        model.unknown_head.to("cpu")

    model.eval()

    # --- Patch forward to handle cross-device tensor movement ---
    gpu_device = torch.device(device)
    hybrid_forward = _HybridInterpretableForward(model, gpu_device)
    model.forward = hybrid_forward  # type: ignore[assignment]

    # Log device placement
    gpu_vram = _estimate_vram_mb(model, device)
    cpu_ram = _estimate_vram_mb(model, "cpu")
    logger.info(f"GPU VRAM: ~{gpu_vram:.0f} MB | CPU RAM: ~{cpu_ram:.0f} MB")

    return _finalize_generator(
        model, raw_config, is_interpretable, device, skip_device_move=True
    )


# ---------------------------------------------------------------------------
# Strategy 3: torchao INT4 (PyTorch-native, no bitsandbytes)
# ---------------------------------------------------------------------------


def _apply_torchao_int4(
    module: nn.Module,
    *,
    skip_names: set[str] | None = None,
) -> None:
    """Apply torchao int4 weight-only quantization, skipping named submodules.

    torchao's ``quantize_`` mutates the model in-place, replacing Linear
    layers with quantized equivalents.  We first detach the submodules
    listed in *skip_names*, quantize, then reattach them.
    """
    from torchao.quantization import int4_weight_only, quantize_

    skip_names = skip_names or set()
    stashed: dict[str, nn.Module] = {}

    # Temporarily detach modules we want to skip
    for name in skip_names:
        if hasattr(module, name):
            stashed[name] = getattr(module, name)
            setattr(module, name, nn.Identity())

    quantize_(module, int4_weight_only())

    # Reattach skipped modules
    for name, child in stashed.items():
        setattr(module, name, child)


def load_torchao(
    model_name_or_path: str = "guidelabs/steerling-8b",
    device: str = "cuda",
) -> SteerlingGenerator:
    """Load a Steerling model with torchao INT4 weight-only quantization.

    A **bitsandbytes-free** alternative.  Uses PyTorch-native INT4
    quantization from the ``torchao`` package.  VRAM usage is comparable
    to the bnb NF4 path (~5 GB).

    Requires ``torchao`` (``pip install torchao``).

    Args:
        model_name_or_path: HuggingFace repo ID or local directory.
        device: Target device (``"cuda"`` or ``"cpu"``).

    Returns:
        SteerlingGenerator ready for inference.
    """
    try:
        import torchao  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "torchao is required for INT4 quantization. Install it with: pip install torchao"
        ) from err

    from steerling.configs.causal_diffusion import CausalDiffusionConfig

    logger.info(f"Loading {model_name_or_path} with torchao INT4 quantization...")

    model, raw_config, is_interpretable = _load_model_and_config(model_name_or_path)

    model_fields = set(CausalDiffusionConfig.model_fields.keys())
    model_data = {k: v for k, v in raw_config.items() if k in model_fields}
    model_config = CausalDiffusionConfig.model_validate(model_data)

    _break_weight_tying(model, model_config)

    # Move to device first — torchao quantizes on-device
    logger.info(f"Moving to {device}...")
    model = model.to(device)

    # Quantize, skipping concept heads (same reason as bnb path)
    logger.info("Applying torchao INT4 weight-only quantization...")
    _apply_torchao_int4(
        model,
        skip_names={"known_head", "unknown_head"},
    )

    model.eval()

    vram = _estimate_vram_mb(model, device)
    logger.info(f"Estimated VRAM: {vram:.0f} MB")

    return _finalize_generator(model, raw_config, is_interpretable, device)
