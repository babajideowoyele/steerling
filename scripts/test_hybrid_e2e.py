"""End-to-end test: load_hybrid with real model weights.

Run: python scripts/test_hybrid_e2e.py
"""

import gc
import logging
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    from steerling.quantization import load_hybrid
    from steerling.configs.generation import GenerationConfig

    # --- Load hybrid ---
    logger.info("=" * 60)
    logger.info("TASK 1: load_hybrid end-to-end")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    generator = load_hybrid("guidelabs/steerling-8b", device="cuda")
    load_time = time.perf_counter() - t0
    logger.info(f"Load time: {load_time:.1f}s")

    # Verify device placement
    model = generator.model
    transformer_devices = {p.device.type for p in model.transformer.parameters()}
    known_devices = {p.device.type for p in model.known_head.parameters()}
    unknown_devices = (
        {p.device.type for p in model.unknown_head.parameters()}
        if model.unknown_head is not None
        else set()
    )

    logger.info(f"Transformer devices: {transformer_devices}")
    logger.info(f"Known head devices: {known_devices}")
    logger.info(f"Unknown head devices: {unknown_devices}")

    assert "cuda" in transformer_devices, "Transformer should be on CUDA"
    assert known_devices == {"cpu"}, f"Known head should be on CPU, got {known_devices}"
    if unknown_devices:
        assert unknown_devices == {"cpu"}, f"Unknown head should be on CPU, got {unknown_devices}"

    logger.info("Device placement: OK")

    # VRAM measurement
    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    vram_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    logger.info(f"VRAM allocated: {vram_mb:.0f} MB | reserved: {vram_reserved_mb:.0f} MB")

    # --- Generate ---
    logger.info("-" * 60)
    logger.info("Generating 50 tokens...")

    config = GenerationConfig(max_new_tokens=50, seed=42)
    prompt = "The future of artificial intelligence"

    t0 = time.perf_counter()
    output = generator.generate_full(prompt, config)
    gen_time = time.perf_counter() - t0

    logger.info(f"Prompt: {prompt}")
    logger.info(f"Generated ({output.generated_tokens} tokens, {gen_time:.1f}s):")
    logger.info(f"  {output.text}")

    # Peak VRAM after generation
    vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    logger.info(f"VRAM peak: {vram_peak_mb:.0f} MB")

    # --- Clean up hybrid, prepare for quantized benchmark ---
    del generator, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- Benchmark quantized for comparison ---
    logger.info("=" * 60)
    logger.info("TASK 2: load_quantized benchmark for comparison")
    logger.info("=" * 60)

    from steerling.quantization import load_quantized

    t0 = time.perf_counter()
    generator_q = load_quantized("guidelabs/steerling-8b", device="cuda")
    load_time_q = time.perf_counter() - t0
    logger.info(f"Load time: {load_time_q:.1f}s")

    vram_q_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    logger.info(f"VRAM allocated: {vram_q_mb:.0f} MB")

    t0 = time.perf_counter()
    output_q = generator_q.generate_full(prompt, config)
    gen_time_q = time.perf_counter() - t0

    logger.info(f"Generated ({output_q.generated_tokens} tokens, {gen_time_q:.1f}s):")
    logger.info(f"  {output_q.text}")

    vram_peak_q_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    logger.info(f"VRAM peak: {vram_peak_q_mb:.0f} MB")

    # --- Clean up quantized, prepare for steering ---
    del generator_q
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- Concept steering ---
    logger.info("=" * 60)
    logger.info("TASK 3: Concept steering with load_quantized")
    logger.info("=" * 60)

    generator_s = load_quantized("guidelabs/steerling-8b", device="cuda")

    prompt_steer = "Communication between people from different cultures"

    # Baseline (no steering)
    config_steer = GenerationConfig(max_new_tokens=50, seed=123)

    logger.info(f"Prompt: {prompt_steer}")
    logger.info("-" * 40)

    t0 = time.perf_counter()
    baseline = generator_s.generate_full(prompt_steer, config_steer)
    t_baseline = time.perf_counter() - t0
    logger.info(f"Baseline ({t_baseline:.1f}s): {baseline.text}")

    # Steer known concepts — try a few concept IDs with positive/negative weights
    # Concept IDs are indices into the 34K known concept codebook
    for concept_id, weight, label in [
        (100, 2.0, "amplify concept 100"),
        (100, -2.0, "suppress concept 100"),
        (500, 3.0, "amplify concept 500"),
        (1000, 3.0, "amplify concept 1000"),
    ]:
        config_steered = GenerationConfig(
            max_new_tokens=50,
            seed=123,
            steer_known={concept_id: weight},
        )
        t0 = time.perf_counter()
        steered = generator_s.generate_full(prompt_steer, config_steered)
        t_steer = time.perf_counter() - t0
        logger.info(f"Steered [{label}] ({t_steer:.1f}s): {steered.text}")

    logger.info("=" * 60)
    logger.info("ALL TASKS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
