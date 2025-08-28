#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stepAnswer_generate_gpt4_test3.py

Reads a JSON array of items:
[
  {
    "QuestionID": "...",
    "Question": "...",
    "RetrievedPassages": ["...", "..."],
    "Answer": ""
  },
  ...
]

Generates GPT-4 answers for the first N items (default 3) and writes ONLY the
items that received an answer during this run to <input_basename>_answer.json.

Usage:
  export OPENAI_API_KEY=sk-...
  python src/stepAnswer_generate_gpt4_test3.py \
    --input outputs/answers/bm25_test.json \
    --limit 3 \
    --model gpt-4.1 \
    --temperature 0.0
"""

import os
import re
import json
import argparse
from typing import List

try:
    from openai import OpenAI, APIError, APIConnectionError, APIStatusError  # type: ignore
except Exception as e:
    raise SystemExit(
        "The OpenAI Python SDK is required. Install with:\n"
        "  pip install --upgrade openai\n\n"
        f"Import error: {e}"
    )

# --- Improved system prompt (obligations + no contradictions) ---
SYSTEM_INSTRUCTIONS = (
    "You are a careful compliance QA assistant. Your job is to extract and present ALL obligations "
    "that are directly supported by the retrieved passages for the given question.\n\n"
    "Rules:\n"
    "1) Use ONLY the provided passages—no outside knowledge or speculation.\n"
    "2) Include every obligation explicitly supported and relevant to the question.\n"
    "3) Each obligation MUST cite its evidence as [P#] (one or more), using only valid passage indices.\n"
    "4) If the passages are incomplete, ambiguous, or contain contradictory obligations relevant to the question, "
    "   reply exactly with: Insufficient evidence in retrieved passages.\n"
    "5) Keep it concise and factual. Use a short bullet list; avoid repetition.\n"
    "6) Preserve modality (must/shall/should); do not change meanings.\n"
    "7) Do not include obligations that are outside the scope of the question, even if present in the passages."
)



def build_user_input(question: str, passages: list[str]) -> str:
    parts = []
    parts.append("Question:")
    parts.append(question.strip())
    parts.append("")
    parts.append("Retrieved Passages:")
    if not passages:
        parts.append("(none)")
    else:
        for i, p in enumerate(passages, 1):
            text = (p or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            parts.append(f"[P{i}] {text}")
    parts.append("")
    parts.append(
        "Task: Based ONLY on the passages above, list ALL obligations that answer the question. "
        "Each bullet MUST end with supporting [P#] cites. "
        "If evidence is insufficient or contradictory, reply exactly: "
        "'Insufficient evidence in retrieved passages.'"

    )
    return "\n".join(parts)



def call_openai_answer(client: OpenAI, model: str, question: str, passages: List[str],
                       temperature: float = 0.0, max_output_tokens: int = 300) -> str:
    user_input = build_user_input(question, passages)
    resp = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=user_input,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return (resp.output_text or "").strip()

def derive_output_path(input_path: str) -> str:
    # e.g., outputs/answers/bm25_test.json -> outputs/answers/bm25_test_answer.json
    if input_path.endswith(".json"):
        return re.sub(r"\.json$", "_answer.json", input_path)
    return input_path + "_answer.json"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSON (e.g., outputs/answers/bm25_test.json)")
    ap.add_argument("--limit", type=int, default=10000, help="Max number of items to answer (default: 10000)")
    ap.add_argument("--model", default="gpt-4.1", help="OpenAI model (default: gpt-4.1)")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    ap.add_argument("--max-output-tokens", type=int, default=300, help="Max tokens in the answer (default: 300)")
    ap.add_argument("--only-missing", action="store_true",
                    help="If set, only answer items whose 'Answer' field is empty.")
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
        raise SystemExit("Please set OPENAI_API_KEY in your environment.")

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit("Input JSON must be an array of items.")

    client = OpenAI()

    # Determine which indices to process (first N, optionally only those missing answers)
    first_n = min(args.limit, len(data))
    indices = list(range(first_n))
    if args.only_missing:
        indices = [i for i in indices if not (data[i].get("Answer") or "").strip()]

    print(f"[INFO] Will attempt answers for {len(indices)} item(s) out of first {first_n} entries.")

    answered_items = []  # collect ONLY items we answered now

    for count, i in enumerate(indices, 1):
        item = data[i]
        qid = item.get("QuestionID", "")
        question = (item.get("Question") or "").strip()
        passages = item.get("RetrievedPassages") or []

        if not question:
            print(f"[SKIP] QID={qid} missing question text.")
            continue

        try:
            answer = call_openai_answer(
                client=client,
                model=args.model,
                question=question,
                passages=passages,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )
        except (APIConnectionError, APIStatusError, APIError) as e:
            print(f"[WARN] QID={qid} OpenAI error: {e}. Skipping.")
            continue

        # Consider any non-empty string as "generated"
        if answer.strip():
            # Make a shallow copy to avoid mutating the original data list
            out_item = {
                "QuestionID": item.get("QuestionID"),
                "Question": item.get("Question"),
                "RetrievedPassages": item.get("RetrievedPassages", []),
                "Answer": answer.strip(),
            }
            answered_items.append(out_item)
            print(f"[OK] ({count}/{len(indices)}) QID={qid} answered.")
        else:
            print(f"[SKIP] ({count}/{len(indices)}) QID={qid} produced empty answer.")

    out_path = derive_output_path(args.input)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(answered_items, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved {len(answered_items)} answered item(s) -> {out_path}")

if __name__ == "__main__":
    main()
