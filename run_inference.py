# Inference Run — Gemma 4 E2B base + instruction-tuned
#
# Generates responses from both models for every prompt in the eval set
# (data/eval_prompts.jsonl, built by prepare_dataset.py) and saves them to
# data/responses.json for the judging script.
#
# Generation is greedy (do_sample=False) so runs are reproducible.
# The output file is rewritten after every batch, so a crash loses at most
# one batch of work.

import gc
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

PROMPTS_PATH = Path(os.environ.get("INF_PROMPTS", "data/eval_prompts.jsonl"))
OUTPUT_PATH = Path(os.environ.get("INF_OUTPUT", "data/responses.json"))
BATCH_SIZE = int(os.environ.get("INF_BATCH_SIZE", 32))
MAX_NEW_TOKENS = int(os.environ.get("INF_MAX_NEW_TOKENS", 256))

MODELS = {
    "base": "google/gemma-4-E2B",
    "it": "google/gemma-4-E2B-it",
}
SYSTEM_PROMPT = "You are a helpful assistant."


def build_texts(processor, prompts, chat):
    """Base model gets the raw prompt (plain completion); IT model gets the chat template."""
    if not chat:
        return prompts
    return [
        processor.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]


def save(results):
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def run_model(tag, model_id, prompt_records, results):
    print(f"\n=== {tag}: {model_id} ===")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    tok = getattr(processor, "tokenizer", processor)
    tok.padding_side = "left"  # required for correct batched generation
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
    print(f"load time: {time.perf_counter() - t0:.1f}s")

    chat = tag == "it"
    for start in range(0, len(prompt_records), BATCH_SIZE):
        batch = prompt_records[start : start + BATCH_SIZE]
        texts = build_texts(processor, [r["prompt"] for r in batch], chat)
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        dt = time.perf_counter() - t0

        responses = tok.batch_decode(out[:, input_len:], skip_special_tokens=True)
        for rec, resp in zip(batch, responses):
            results.append(
                {
                    "id": rec["id"],
                    "model": tag,
                    "model_id": model_id,
                    "language": rec["language"],
                    "category": rec["category"],
                    "prompt": rec["prompt"],
                    "response": resp.strip(),
                    "max_new_tokens": MAX_NEW_TOKENS,
                }
            )
        save(results)
        done = min(start + BATCH_SIZE, len(prompt_records))
        print(f"  {done}/{len(prompt_records)} prompts ({dt:.1f}s/batch)")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def main():
    assert torch.cuda.is_available(), "No GPU visible!"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    prompt_records = [json.loads(line) for line in PROMPTS_PATH.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(prompt_records)} prompts from {PROMPTS_PATH}")

    results = []
    t0 = time.perf_counter()
    for tag, model_id in MODELS.items():
        run_model(tag, model_id, prompt_records, results)

    print(f"\nWrote {len(results)} responses to {OUTPUT_PATH}")
    print(f"Total time: {time.perf_counter() - t0:.0f}s")


main()
