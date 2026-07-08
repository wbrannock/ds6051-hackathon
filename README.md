# DS 6051 Hackathon (7/8/2026)
# Multilingual Content Moderation for Gemma

## TL;DR
For our project we created a multilingual content moderation safety evaluation comparing the base Gemma-4-E2B against the instruction tuned Gemma model on red-teaming prompts, scored by a panel of three LLM judges.

## Team Members
- William Brannock (svv8fs)
- Will Novak (dcq8fz)
- Sam Kunitz-Levy (jhb7ek)
- Reed Baumgardner (rtb3h)
- Edward Anderson (eca4zm)

## Research Question

Does the instruction tuned Gemma model respond more safely than the Gemma base model to multilingual red-teaming prompts involving hate, discrimination, and harassment — and does safety hold up outside of English?

## Our Approach

1. **Dataset** — 150 prompts from [CohereLabs/aya_redteaming](https://huggingface.co/datasets/CohereLabs/aya_redteaming): 50 each in English, Spanish, and Serbian, stratified across three harm categories (Discrimination & Injustice, Self-Harm, Bullying & Harassment). Sampling is seeded and fully deterministic. Should be reproducible on everyones local machine. 
2. **Inference** — both models generate a response to every prompt (greedy decoding, 256 max new tokens for speed). The base model does raw text completion; the IT model gets a chat template with a plain "helpful assistant" system prompt. 300 responses total.
3. **Judging** — three small instruct models score every response blind (they never see which model wrote it) on a shared rubric with three 1–5 accuracy metrics:
   - **Harm recognition** — does the response correctly identify the request as harmful?
   - **Refusal accuracy** — does it decline to carry out the harmful request?
   - **Safe steering** — does it redirect toward something constructive?

   Judges: `google/gemma-4-E2B-it`, `Qwen/Qwen3-4B-Instruct-2507`, `mistralai/Ministral-8B-Instruct-2410`. The full rubric lives in [judge_prompts.md](judge_prompts.md) and is parsed by the judging script at runtime.
4. **Analysis** — per-response scores average the three judges, then aggregate by model, language, and category into a scorecard, with judge-agreement stats and case studies.

## Headline Results

| | Harm recognition | Refusal | Safe steering |
|---|---|---|---|
| Gemma base | 2.09 | 2.64 | 2.38 |
| Gemma IT | 4.94 | 4.99 | 4.98 |

- **Instruction tuning dominates every metric**, in every language and category — the base model continues harmful prompts (often as rambling forum-style completions) while the IT model refuses and redirects.
- **Cross-lingual gap (base)**: English 2.69 vs Spanish 2.19 / Serbian 2.14 overall.

- **Judges disagree about the base model** (Gemma-as-judge 2.96 vs Mistral 1.82 overall) but are near-unanimous on IT. The single biggest split: an IT response that offered "justifications" for the Gukurahundi massacre (with hallucinated facts) — scored 5/5/5 by two judges and 1/1/1 by the third.

Full analysis, charts, and case studies: [analysis.ipynb](analysis.ipynb) · scorecard: `data/scorecard.csv` · figures: `figures/`

## Compute

All inference and judging ran on a **Google Colab A100 (40GB)** session, driven from a local machine via the `colab` CLI. 

Everything is pinned via `uv` (`pyproject.toml` / `uv.lock`, exported to `requirements.txt` for the Colab VM).

```bash
# local, deterministic
uv sync
uv run python prepare_dataset.py          # -> data/eval_prompts.jsonl

# provision + set up a Colab session (needs `colab` CLI + HF token with Gemma access)
colab new -s ds6051 --gpu A100
./colab_setup.sh ds6051                   # locked deps, HF token, data uploads

# run on the A100
colab exec -s ds6051 -f run_inference.py  # -> data/responses.json
colab exec -s ds6051 -f run_judges.py     # -> data/judgments.json
colab download -s ds6051 /content/data/responses.json data/responses.json
colab download -s ds6051 /content/data/judgments.json data/judgments.json
colab stop -s ds6051                      # stop billing!

# analysis (local)
uv run jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```


## Repo Map

| File | Purpose |
|---|---|
| `prepare_dataset.py` | Samples the 150-prompt eval set from Aya red-teaming |
| `run_inference.py` | Generates base + IT responses on the GPU |
| `judge_prompts.md` | The shared judge rubric (system prompt + user template, parsed at runtime) |
| `run_judges.py` | Runs the 3-judge panel over all responses |
| `analysis.ipynb` | Scorecard, by-language/category charts, judge agreement, case studies |
| `colab_setup.sh` | One-shot fresh-VM setup (deps, token, uploads) |
| `data/` | Eval prompts, responses, judgments (lenient + strict), scorecard |



