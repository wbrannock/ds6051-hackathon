# LLM-as-Judge Run — three judges over data/responses.json
#
# Judges every (prompt, response) record with three small instruct models,
# using the shared rubric defined in judge_prompts.md (the system prompt and
# user template are parsed from that file — edit the markdown, not this
# script, to change the rubric).
#
# Judges run sequentially (one model in VRAM at a time). Output is
# data/judgments.json — one flat record per (response x judge) with the three
# 1-5 scores. The file is rewritten after every batch, so a crash loses at
# most one batch. Records that fail JSON parsing keep the raw judge text for
# debugging and get scores=None.
#
# Generation is greedy (do_sample=False) so judge scores are reproducible.

import gc
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

RESPONSES_PATH = Path(os.environ.get("JUDGE_INPUT", "data/responses.json"))
OUTPUT_PATH = Path(os.environ.get("JUDGE_OUTPUT", "data/judgments.json"))
PROMPTS_MD = Path(os.environ.get("JUDGE_PROMPTS_MD", "judge_prompts.md"))
BATCH_SIZE = int(os.environ.get("JUDGE_BATCH_SIZE", 16))
MAX_NEW_TOKENS = int(os.environ.get("JUDGE_MAX_NEW_TOKENS", 96))

JUDGES = {
    "gemma": "google/gemma-4-E2B-it",
    "qwen": "Qwen/Qwen3-4B-Instruct-2507",
    "mistral": "mistralai/Ministral-8B-Instruct-2410",
}

METRICS = ["harm_accuracy_recognition", "refusal_accuracy", "safe_steering_accuracy"]


def load_rubric():
    """The first ```text block in judge_prompts.md is the system prompt, the second is the user template."""
    blocks = re.findall(r"```text\n(.*?)```", PROMPTS_MD.read_text(), re.DOTALL)
    if len(blocks) < 2:
        raise ValueError(f"Expected 2 ```text blocks in {PROMPTS_MD}, found {len(blocks)}")
    system_prompt, user_template = blocks[0].strip(), blocks[1].strip()
    for placeholder in ("{prompt}", "{response}"):
        if placeholder not in user_template:
            raise ValueError(f"User template is missing {placeholder}")
    return system_prompt, user_template


def build_chat_text(processor, system_prompt, user_msg):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        # Some chat templates reject a system role — fold it into the user turn.
        merged = [{"role": "user", "content": f"{system_prompt}\n\n{user_msg}"}]
        return processor.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )


def parse_scores(text):
    """Extract the first JSON object with all three metrics; None if unparseable."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    scores = {}
    for metric in METRICS:
        value = obj.get(metric)
        if not isinstance(value, (int, float)) or not 1 <= value <= 5:
            return None
        scores[metric] = int(value)
    return scores


def save(results):
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def run_judge(judge_tag, judge_id, records, system_prompt, user_template, results):
    done = {(r["id"], r["model"]) for r in results if r["judge"] == judge_tag}
    records = [r for r in records if (r["id"], r["model"]) not in done]
    if not records:
        print(f"\n=== judge: {judge_tag} — already complete, skipping ===")
        return 0

    print(f"\n=== judge: {judge_tag} ({judge_id}) — {len(records)} to judge ===")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(judge_id)
    tok = getattr(processor, "tokenizer", processor)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # e.g. Mistral tokenizers ship without one
    model = AutoModelForCausalLM.from_pretrained(judge_id, dtype="auto", device_map="auto")
    print(f"load time: {time.perf_counter() - t0:.1f}s")

    failures = 0
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        texts = [
            build_chat_text(
                processor,
                system_prompt,
                user_template.format(prompt=rec["prompt"], response=rec["response"]),
            )
            for rec in batch
        ]
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        dt = time.perf_counter() - t0

        raw_outputs = tok.batch_decode(out[:, input_len:], skip_special_tokens=True)
        for rec, raw in zip(batch, raw_outputs):
            scores = parse_scores(raw)
            if scores is None:
                failures += 1
            results.append(
                {
                    "id": rec["id"],
                    "model": rec["model"],  # subject model (base / it)
                    "judge": judge_tag,
                    "judge_id": judge_id,
                    "language": rec["language"],
                    "category": rec["category"],
                    "scores": scores,
                    **({} if scores is not None else {"raw": raw.strip()}),
                }
            )
        save(results)
        done = min(start + BATCH_SIZE, len(records))
        print(f"  {done}/{len(records)} responses ({dt:.1f}s/batch, parse failures so far: {failures})")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return failures


def main():
    assert torch.cuda.is_available(), "No GPU visible!"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    system_prompt, user_template = load_rubric()
    print(f"Rubric loaded from {PROMPTS_MD} (system prompt: {len(system_prompt)} chars)")

    records = json.loads(RESPONSES_PATH.read_text())
    print(f"Loaded {len(records)} responses from {RESPONSES_PATH}")

    results = json.loads(OUTPUT_PATH.read_text()) if OUTPUT_PATH.exists() else []
    if results:
        print(f"Resuming: {len(results)} existing judgments in {OUTPUT_PATH}")
    t0 = time.perf_counter()
    total_failures = 0
    for judge_tag, judge_id in JUDGES.items():
        total_failures += run_judge(judge_tag, judge_id, records, system_prompt, user_template, results)

    print(f"\nWrote {len(results)} judgments to {OUTPUT_PATH}")
    print(f"Parse failures: {total_failures} (records with scores=null, raw text kept)")
    print(f"Total time: {time.perf_counter() - t0:.0f}s")


main()
