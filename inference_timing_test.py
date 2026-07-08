# Inference Timing Test — Gemma 4 E2B (base + it)
#
# Measures single-prompt latency and batched throughput for both models,
# then extrapolates to the full ToxiGen run so we can budget generation time.
# Prints timings only — never the generated text.
#
# Runs as-is in a Jupyter/Colab kernel or via `python inference_timing_test.py`.
# Configure via environment variables (TT_* below).

import gc
import os
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MAX_NEW_TOKENS = int(os.environ.get("TT_MAX_NEW_TOKENS", 256))
N_SINGLE = int(os.environ.get("TT_N_SINGLE", 4))        # single-prompt timing runs
BATCH_SIZE = int(os.environ.get("TT_BATCH_SIZE", 8))    # batched timing
N_BATCHES = int(os.environ.get("TT_N_BATCHES", 2))
PROMPTS_FILE = os.environ.get("TT_PROMPTS_FILE", "/content/toxigen_sample.txt")
TOTAL_PROMPTS = int(os.environ.get("TT_TOTAL_PROMPTS", 1000))  # extrapolation target

MODELS = {
    "base": "google/gemma-4-E2B",
    "it": "google/gemma-4-E2B-it",
}

FALLBACK_PROMPTS = [
    "- the museum opens at nine on weekdays\\n- the gift shop closes early on sundays\\n- "
    "parking is free after six in the evening\\n- the cafe is on the second floor\\n-",
    "- the bus route changed last month\\n- the new schedule adds two morning stops\\n- "
    "riders can check arrival times in the app\\n- monthly passes went up a dollar\\n-",
] * 4


def load_prompts():
    p = Path(PROMPTS_FILE)
    if p.exists():
        lines = [line.strip() for line in p.read_text().splitlines() if line.strip()]
        if lines:
            print(f"Loaded {len(lines)} prompts from {p}")
            return lines
    print("Prompts file not found; using neutral fallback prompts")
    return FALLBACK_PROMPTS


def prepare_inputs(processor, prompts, chat):
    if chat:
        texts = [
            processor.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": p},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in prompts
        ]
    else:
        texts = prompts
    return processor(text=texts, return_tensors="pt", padding=True)


def timed_generate(model, processor, prompts, chat):
    """Returns (wall_seconds, generated_token_count) for one generate() call."""
    inputs = prepare_inputs(processor, prompts, chat).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gen = out[:, input_len:]
    tok = getattr(processor, "tokenizer", processor)
    pad_id = tok.pad_token_id
    n_tokens = int((gen != pad_id).sum()) if pad_id is not None else gen.numel()
    return dt, n_tokens


def bench(tag, model_id, prompts):
    print(f"\n=== {tag}: {model_id} ===")
    chat = tag == "it"

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    tok = getattr(processor, "tokenizer", processor)
    tok.padding_side = "left"  # required for correct batched generation
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
    print(f"load time: {time.perf_counter() - t0:.1f}s")

    print("warmup...")
    timed_generate(model, processor, prompts[:1], chat)

    single_times, single_tokens = [], []
    for i in range(N_SINGLE):
        dt, n = timed_generate(model, processor, [prompts[i % len(prompts)]], chat)
        single_times.append(dt)
        single_tokens.append(n)
        print(f"  single {i + 1}/{N_SINGLE}: {dt:.2f}s, {n} tokens ({n / dt:.1f} tok/s)")

    batch_times, batch_tokens = [], []
    for b in range(N_BATCHES):
        start = (b * BATCH_SIZE) % max(1, len(prompts) - BATCH_SIZE + 1)
        batch = prompts[start : start + BATCH_SIZE]
        dt, n = timed_generate(model, processor, batch, chat)
        batch_times.append(dt)
        batch_tokens.append(n)
        print(
            f"  batch {b + 1}/{N_BATCHES} (size {len(batch)}): {dt:.2f}s, "
            f"{n} tokens ({n / dt:.1f} tok/s, {dt / len(batch):.2f}s/prompt)"
        )

    med_single = statistics.median(single_times)
    per_prompt_batched = statistics.median(batch_times) / BATCH_SIZE
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    result = {
        "median_single_s": round(med_single, 2),
        "per_prompt_batched_s": round(per_prompt_batched, 2),
        "throughput_tok_s": round(statistics.median(t / d for d, t in zip(batch_times, batch_tokens)), 1),
        "peak_vram_gb": round(peak_gb, 1),
        "est_minutes_per_1k_prompts_batched": round(per_prompt_batched * 1000 / 60, 1),
    }
    print(f"{tag} summary: {result}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    assert torch.cuda.is_available(), "No GPU visible!"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"config: max_new_tokens={MAX_NEW_TOKENS}, batch_size={BATCH_SIZE}")

    prompts = load_prompts()
    results = {tag: bench(tag, mid, prompts) for tag, mid in MODELS.items()}

    print("\n=== EXTRAPOLATION ===")
    total_min = 0.0
    for tag, r in results.items():
        mins = r["per_prompt_batched_s"] * TOTAL_PROMPTS / 60
        total_min += mins
        print(f"{tag}: {TOTAL_PROMPTS} prompts @ batch {BATCH_SIZE} ≈ {mins:.0f} min")
    print(f"both models, {TOTAL_PROMPTS} prompts each ≈ {total_min:.0f} min total")
    print("(scales linearly: 2x prompts → 2x time; larger batches will be faster)")


main()
