"""Tests for 4-bit quantization and hybrid offloading.

These tests use tiny model configs. Structural tests (layer replacement,
weight tying, device placement) run on CPU. Forward pass tests that go
through bnb.nn.Linear4bit require CUDA and are skipped otherwise.
"""

import pytest
import torch
import torch.nn as nn

from steerling.configs.causal_diffusion import CausalDiffusionConfig
from steerling.configs.concept import ConceptConfig
from steerling.models.causal_diffusion import CausalDiffusionLM
from steerling.models.interpretable.interpretable_causal_diffusion import (
    InterpretableCausalDiffusionLM,
)

# Skip entire module if bitsandbytes is not installed
bnb = pytest.importorskip("bitsandbytes")

from steerling.quantization import (  # noqa: E402
    _break_weight_tying,
    _replace_linear_with_4bit,
    _HybridInterpretableForward,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for bnb 4-bit forward"
)


# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def tiny_config() -> CausalDiffusionConfig:
    return CausalDiffusionConfig(
        n_layers=2,
        n_head=4,
        n_embd=128,
        block_size=256,
        n_kv_heads=2,
        diff_block_size=16,
        use_rms_norm=True,
        norm_order="post",
        use_qk_norm=True,
        use_rope=True,
        rope_base=500000.0,
        mlp_type="swiglu",
        use_bias=False,
        clip_qkv=10.0,
        weight_sharing=True,
    )


@pytest.fixture
def tiny_concept_config() -> ConceptConfig:
    return ConceptConfig(
        n_concepts=32,
        n_unknown_concepts=64,
        concept_dim=128,
        use_attention_known=False,
        use_attention_unknown=False,
        topk_known=4,
        topk_known_features=4,
        unknown_topk=8,
        use_unknown=True,
        factorize_unknown=False,
        use_epsilon_correction=True,
        block_size=256,
        pad_multiple=16,
        apply_topk_to_unknown=True,
        inject_layer=1,
    )


@pytest.fixture
def vocab_size() -> int:
    return 256


@pytest.fixture
def base_model(tiny_config, vocab_size):
    model = CausalDiffusionLM(tiny_config, vocab_size=vocab_size)
    return model.to(dtype=torch.bfloat16)


@pytest.fixture
def interpretable_model(tiny_config, tiny_concept_config, vocab_size):
    model = InterpretableCausalDiffusionLM(
        config=tiny_config,
        concept_config=tiny_concept_config,
        vocab_size=vocab_size,
    )
    return model.to(dtype=torch.bfloat16)


# ---- Helper ------------------------------------------------------------------


def _count_linear_types(model: nn.Module) -> dict[str, int]:
    """Count nn.Linear vs bnb.nn.Linear4bit in model."""
    counts: dict[str, int] = {"Linear": 0, "Linear4bit": 0}
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            counts["Linear4bit"] += 1
        elif isinstance(module, nn.Linear):
            counts["Linear"] += 1
    return counts


# ---- Tests: _replace_linear_with_4bit (CPU) ----------------------------------


class TestReplaceLinear:
    def test_all_linears_replaced(self, base_model):
        before = _count_linear_types(base_model)
        assert before["Linear"] > 0
        assert before["Linear4bit"] == 0

        _replace_linear_with_4bit(base_model)

        after = _count_linear_types(base_model)
        assert after["Linear"] == 0
        assert after["Linear4bit"] == before["Linear"]

    def test_skip_names(self, base_model):
        _replace_linear_with_4bit(base_model, skip_names={"lm_head"})

        assert isinstance(base_model.lm_head, nn.Linear)
        assert not isinstance(base_model.lm_head, bnb.nn.Linear4bit)

    def test_embeddings_untouched(self, base_model):
        _replace_linear_with_4bit(base_model)
        assert isinstance(base_model.tok_emb, nn.Embedding)

    def test_skip_concept_heads(self, interpretable_model):
        _replace_linear_with_4bit(
            interpretable_model,
            skip_names={"known_head", "unknown_head"},
        )

        # Concept heads should still have nn.Linear
        known_linears = [
            m
            for m in interpretable_model.known_head.modules()
            if isinstance(m, nn.Linear) and not isinstance(m, bnb.nn.Linear4bit)
        ]
        assert len(known_linears) > 0

        # Transformer should be quantized
        transformer_4bit = [
            m for m in interpretable_model.transformer.modules() if isinstance(m, bnb.nn.Linear4bit)
        ]
        assert len(transformer_4bit) > 0

    def test_weight_shapes_preserved(self, base_model):
        """4-bit replacement preserves in/out features."""
        shapes_before: dict[str, tuple[int, int]] = {}
        for name, m in base_model.named_modules():
            if isinstance(m, nn.Linear):
                shapes_before[name] = (m.in_features, m.out_features)

        _replace_linear_with_4bit(base_model)

        for name, m in base_model.named_modules():
            if isinstance(m, bnb.nn.Linear4bit):
                assert (m.in_features, m.out_features) == shapes_before[name]


# ---- Tests: Weight tying (CPU) -----------------------------------------------


class TestWeightTying:
    def test_tying_broken(self, base_model, tiny_config):
        assert base_model.tok_emb.weight.data_ptr() == base_model.lm_head.weight.data_ptr()

        _break_weight_tying(base_model, tiny_config)

        assert base_model.tok_emb.weight.data_ptr() != base_model.lm_head.weight.data_ptr()
        assert torch.equal(base_model.tok_emb.weight.data, base_model.lm_head.weight.data)

    def test_tying_not_broken_when_disabled(self, vocab_size):
        config = CausalDiffusionConfig(
            n_layers=2,
            n_head=4,
            n_embd=128,
            block_size=256,
            n_kv_heads=2,
            diff_block_size=16,
            weight_sharing=False,
        )
        model = CausalDiffusionLM(config, vocab_size=vocab_size).to(dtype=torch.bfloat16)
        ptr_before = model.lm_head.weight.data_ptr()

        _break_weight_tying(model, config)

        assert model.lm_head.weight.data_ptr() == ptr_before


# ---- Tests: Hybrid forward (CPU, no quantization) ----------------------------
# Tests the device-shuttling logic of _HybridInterpretableForward using
# plain bf16 models (no bnb quantization), so they run on CPU.


class TestHybridForward:
    def test_minimal_output(self, interpretable_model, tiny_config, vocab_size):
        """Hybrid forward produces valid logits with minimal_output=True."""
        _break_weight_tying(interpretable_model, tiny_config)
        interpretable_model.eval()

        cpu = torch.device("cpu")
        hybrid_fwd = _HybridInterpretableForward(interpretable_model, gpu_device=cpu)

        B, T = 1, 32
        input_ids = torch.randint(0, vocab_size, (B, T))

        with torch.no_grad():
            logits, outputs = hybrid_fwd(input_ids, minimal_output=True)

        assert logits.shape == (B, T, vocab_size)
        assert torch.isfinite(logits).all()

    def test_full_output(self, interpretable_model, tiny_config, vocab_size):
        """Hybrid forward produces full attribution output."""
        _break_weight_tying(interpretable_model, tiny_config)
        interpretable_model.eval()

        cpu = torch.device("cpu")
        hybrid_fwd = _HybridInterpretableForward(interpretable_model, gpu_device=cpu)

        B, T = 1, 16
        input_ids = torch.randint(0, vocab_size, (B, T))

        with torch.no_grad():
            logits, outputs = hybrid_fwd(input_ids, minimal_output=False)

        assert logits.shape == (B, T, vocab_size)
        assert outputs.composed.shape == (B, T, tiny_config.n_embd)
        assert outputs.known_features.shape == (B, T, tiny_config.n_embd)
        assert outputs.hidden.shape == (B, T, tiny_config.n_embd)

    def test_output_matches_original(self, interpretable_model, tiny_config, vocab_size):
        """Hybrid forward produces same results as original forward (on same device)."""
        _break_weight_tying(interpretable_model, tiny_config)
        interpretable_model.eval()

        B, T = 1, 16
        input_ids = torch.randint(0, vocab_size, (B, T))

        # Original forward
        with torch.no_grad():
            orig_logits, orig_out = interpretable_model(
                input_ids, use_teacher_forcing=False, minimal_output=True
            )

        # Hybrid forward (both devices = CPU, so should match exactly)
        cpu = torch.device("cpu")
        hybrid_fwd = _HybridInterpretableForward(interpretable_model, gpu_device=cpu)

        with torch.no_grad():
            hybrid_logits, hybrid_out = hybrid_fwd(input_ids, minimal_output=True)

        assert torch.allclose(orig_logits, hybrid_logits, atol=1e-5)


# ---- Tests: Quantized forward (requires CUDA) --------------------------------


def _disable_compiled_flex_attention():
    """Apply the same workaround used in production: disable compiled flex_attention."""
    import steerling.models.layers.causal_diffusion_layers as layers
    from torch.nn.attention.flex_attention import flex_attention

    layers.compiled_flex_attention = flex_attention


class TestQuantizedForwardCUDA:
    @requires_cuda
    def test_base_model_forward(self, base_model, tiny_config, vocab_size):
        _disable_compiled_flex_attention()
        _break_weight_tying(base_model, tiny_config)
        _replace_linear_with_4bit(base_model)
        base_model = base_model.to("cuda")
        base_model.eval()

        B, T = 1, 32
        input_ids = torch.randint(0, vocab_size, (B, T), device="cuda")

        with torch.no_grad():
            logits = base_model(input_ids)

        assert logits.shape == (B, T, vocab_size)
        assert torch.isfinite(logits).all()

    @requires_cuda
    def test_interpretable_model_forward(self, interpretable_model, tiny_config, vocab_size):
        _disable_compiled_flex_attention()
        _break_weight_tying(interpretable_model, tiny_config)
        _replace_linear_with_4bit(
            interpretable_model,
            skip_names={"known_head", "unknown_head"},
        )
        interpretable_model = interpretable_model.to("cuda")
        interpretable_model.eval()

        B, T = 1, 32
        input_ids = torch.randint(0, vocab_size, (B, T), device="cuda")

        with torch.no_grad():
            logits, outputs = interpretable_model(input_ids, use_teacher_forcing=False)

        assert logits.shape == (B, T, vocab_size)
        assert torch.isfinite(logits).all()

    @requires_cuda
    def test_hybrid_forward(self, interpretable_model, tiny_config, vocab_size):
        _disable_compiled_flex_attention()
        _break_weight_tying(interpretable_model, tiny_config)
        _replace_linear_with_4bit(
            interpretable_model,
            skip_names={"known_head", "unknown_head"},
        )
        interpretable_model.transformer.to("cuda")
        interpretable_model.known_head.to("cpu")
        if interpretable_model.unknown_head is not None:
            interpretable_model.unknown_head.to("cpu")
        interpretable_model.eval()

        gpu = torch.device("cuda")
        hybrid_fwd = _HybridInterpretableForward(interpretable_model, gpu_device=gpu)

        B, T = 1, 32
        input_ids = torch.randint(0, vocab_size, (B, T))

        with torch.no_grad():
            logits, outputs = hybrid_fwd(input_ids, minimal_output=True)

        assert logits.shape == (B, T, vocab_size)
        assert logits.device.type == "cuda"
        assert torch.isfinite(logits).all()


# ---- Tests: Device placement -------------------------------------------------


class TestHybridDevicePlacement:
    def test_concept_heads_skipped_from_quantization(self, interpretable_model):
        _replace_linear_with_4bit(
            interpretable_model,
            skip_names={"known_head", "unknown_head"},
        )

        block_linears = _count_linear_types(interpretable_model.transformer)
        assert block_linears["Linear4bit"] > 0

        known_types = _count_linear_types(interpretable_model.known_head)
        assert known_types["Linear4bit"] == 0
        assert known_types["Linear"] > 0
