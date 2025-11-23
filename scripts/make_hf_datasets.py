#!/usr/bin/env python3
"""
Build 3 eval text files (100 prompts each) from Hugging Face datasets:

- Code:        openai_humaneval         -> datasets/code_prompts.txt
- Explain:     tatsu-lab/alpaca         -> datasets/explain_prompts.txt
- Tests/QA:    OpenAssistant/oasst1     -> datasets/tests_prompts.txt

Each line is a single prompt your planner can split/classify.
"""
import random, os
from datasets import load_dataset

os.makedirs("datasets", exist_ok=True)

# 1) Code (HumanEval) — use the natural-language *task description* where possible.
ds_code = load_dataset("openai_humaneval", split="test")
code_prompts = []
for ex in ds_code:
    # Prefer the docstring / prompt text field as a request
    p = (ex.get("prompt") or "").strip()
    if p:
        # Wrap as a clear instruction
        code_prompts.append(f"Implement the following Python function:\n{p}")

random.shuffle(code_prompts)
open("datasets/code_prompts.txt", "w", encoding="utf-8").write("\n".join(code_prompts[:100]) + "\n")

# 2) Explain (Alpaca) — instruction-only is enough
ds_explain = load_dataset("tatsu-lab/alpaca", split="train")
explain_prompts = [ex["instruction"].strip() for ex in ds_explain if ex.get("instruction")]
random.shuffle(explain_prompts)
open("datasets/explain_prompts.txt", "w", encoding="utf-8").write("\n".join(explain_prompts[:100]) + "\n")

# 3) Tests/QA (OpenAssistant) — take prompter messages as tasks to “verify” or “test”
ds_tests = load_dataset("OpenAssistant/oasst1", split="train")
tests_prompts = []
for ex in ds_tests:
    if ex.get("role") == "prompter":
        t = (ex.get("text") or "").strip()
        if t:
            tests_prompts.append(t)
random.shuffle(tests_prompts)
open("datasets/tests_prompts.txt", "w", encoding="utf-8").write("\n".join(tests_prompts[:100]) + "\n")

print("Wrote: datasets/code_prompts.txt, explain_prompts.txt, tests_prompts.txt")
