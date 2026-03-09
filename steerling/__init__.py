"""
Steerling: An interpretable causal diffusion language model with concept steering.
"""

__version__ = "0.1.0"

from steerling.configs import CausalDiffusionConfig, ConceptConfig, GenerationConfig
from steerling.inference import SteerlingGenerator

__all__ = [
    "CausalDiffusionConfig",
    "ConceptConfig",
    "GenerationConfig",
    "SteerlingGenerator",
    "load_quantized",
    "load_hybrid",
    "load_torchao",
]


def load_quantized(
    model_name_or_path: str = "guidelabs/steerling-8b",
    device: str = "cuda",
) -> SteerlingGenerator:
    """Load a Steerling model with 4-bit NF4 quantization. Requires bitsandbytes."""
    from steerling.quantization import load_quantized as _load_quantized

    return _load_quantized(model_name_or_path, device=device)


def load_hybrid(
    model_name_or_path: str = "guidelabs/steerling-8b",
    device: str = "cuda",
) -> SteerlingGenerator:
    """Load with 4-bit transformer on GPU + bf16 concept heads on CPU. Requires bitsandbytes."""
    from steerling.quantization import load_hybrid as _load_hybrid

    return _load_hybrid(model_name_or_path, device=device)


def load_torchao(
    model_name_or_path: str = "guidelabs/steerling-8b",
    device: str = "cuda",
) -> SteerlingGenerator:
    """Load with torchao INT4 quantization. No bitsandbytes needed. Requires torchao."""
    from steerling.quantization import load_torchao as _load_torchao

    return _load_torchao(model_name_or_path, device=device)
