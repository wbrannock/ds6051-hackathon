# Dataset Preparation — Aya Red-teaming (CohereLabs/aya_redteaming)
#
# Builds the evaluation prompt set: English, Spanish, and Serbian prompts from
# the Discrimination & Injustice, Self-Harm, and Bullying & Harassment harm
# categories, sampled deterministically (fixed seed) for reproducibility.
#
# Output: data/eval_prompts.jsonl — one record per prompt with language,
# assigned category, and English translations (where the source provides them).
#
# Run locally: uv run python prepare_dataset.py

import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

SEED = 6051
PROMPTS_PER_LANGUAGE = 50

LANGUAGES = {  # ISO code used in the repo filenames -> label
    "eng": "English",
    "spa": "Spanish",
    "srp": "Serbian",
}
CATEGORIES = [  # exact labels used by the dataset
    "Discrimination & Injustice",
    "Self-Harm",
    "Bullying & Harassment",
]
OUTPUT_PATH = Path("data/eval_prompts.jsonl")


def split_evenly(total: int, buckets: int) -> list[int]:
    """[17, 17, 16] for (50, 3) — earlier buckets get the remainder."""
    base, rem = divmod(total, buckets)
    return [base + (1 if i < rem else 0) for i in range(buckets)]


def load_language(code: str):
    ds = load_dataset(
        "json",
        data_files=f"hf://datasets/CohereLabs/aya_redteaming/aya_{code}.jsonl",
        split="train",
    )
    rows = []
    for row in ds:
        cats = json.loads(row["harm_category"])
        target_cats = [c for c in CATEGORIES if c in cats]
        if target_cats:
            rows.append({**row, "target_categories": target_cats})
    # Sort for determinism independent of source row order; prompts are unique.
    rows.sort(key=lambda r: r["prompt"])
    return rows


def main():
    rng = random.Random(SEED)
    records = []

    for code, language in LANGUAGES.items():
        rows = load_language(code)

        # Assign each row to one bucket (its first matching category) so a
        # multi-label prompt can't be sampled twice.
        buckets = {c: [] for c in CATEGORIES}
        for row in rows:
            buckets[row["target_categories"][0]].append(row)

        quotas = split_evenly(PROMPTS_PER_LANGUAGE, len(CATEGORIES))
        for category, quota in zip(CATEGORIES, quotas):
            pool = buckets[category]
            if len(pool) < quota:
                print(f"WARNING: {language}/{category} has only {len(pool)} prompts (wanted {quota}); taking all")
            sampled = rng.sample(pool, min(quota, len(pool)))
            for row in sampled:
                records.append(
                    {
                        "id": f"{code}_{len(records):04d}",
                        "language": language,
                        "category": category,
                        "all_categories": json.loads(row["harm_category"]),
                        "prompt": row["prompt"],
                        "global_or_local": row["global_or_local"],
                        "literal_translation": row["literal_translation"],
                        "semantic_translation": row["semantic_translation"],
                    }
                )

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} prompts to {OUTPUT_PATH}")
    print("\nBreakdown (language x category):")
    counts = Counter((r["language"], r["category"]) for r in records)
    for (lang, cat), n in sorted(counts.items()):
        print(f"  {lang:10s} {cat:28s} {n}")


main()
