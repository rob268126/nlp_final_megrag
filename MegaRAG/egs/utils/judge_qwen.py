import argparse
import asyncio
import json
import pathlib
import random
import re
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from qwen_llm import qwen_gpt_4o_mini_complete

SYSTEM_PROMPT = "Bạn là giám khảo đánh giá câu trả lời QA tài liệu tiếng Việt."

JUDGE_PROMPT = """Hãy đánh giá câu trả lời của mô hình so với câu trả lời mẫu.

Câu hỏi:
{question}

Câu trả lời mẫu:
{gold_answer}

Câu trả lời mô hình:
{pred_answer}

Yêu cầu:
- Trả về đúng một JSON hợp lệ, không markdown.
- JSON gồm 2 trường: "verdict" và "reason".
- "verdict" chỉ được là "YES" hoặc "NO".
- "YES" nếu câu trả lời mô hình đúng về ngữ nghĩa và đủ ý so với câu trả lời mẫu.
- "NO" nếu sai, thiếu ý quan trọng, hoặc không liên quan.

JSON:
"""


def parse_verdict(resp: str):
    if not resp:
        return "NO", "empty response"

    m = re.search(r"\{.*\}", resp, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            verdict = str(obj.get("verdict", "")).upper().strip()
            reason = str(obj.get("reason", ""))
            if verdict in ["YES", "NO"]:
                return verdict, reason
            if "YES" in verdict:
                return "YES", reason
            if "NO" in verdict:
                return "NO", reason
        except Exception:
            pass

    upper = resp.upper()
    if "YES" in upper:
        return "YES", resp[:300]
    return "NO", resp[:300]


async def judge_item(sem, item):
    async with sem:
        prompt = JUDGE_PROMPT.format(
            question=item["question"],
            gold_answer=item["gold_answer"],
            pred_answer=item["pred_answer"],
        )
        try:
            resp = await qwen_gpt_4o_mini_complete(
                prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=256,
            )
        except Exception as e:
            resp = f"ERROR: {e}"

        verdict, reason = parse_verdict(resp)
        return {
            "index": item["index"],
            "question": item["question"],
            "verdict": verdict,
            "reason": reason,
            "raw_judge_response": resp,
        }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-items", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sample", type=int, default=20)
    args = ap.parse_args()

    with open(args.eval_items, "r", encoding="utf-8") as f:
        eval_items = [json.loads(line) for line in f if line.strip()]

    preds = {}
    with open(args.results, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            preds[rec["index"]] = rec.get("answer", "") if not rec.get("error") else ""

    judge_inputs = []
    for item in eval_items:
        idx = item["index"]
        if idx in preds:
            judge_inputs.append(
                {
                    "index": idx,
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                    "pred_answer": preds[idx],
                }
            )

    random.seed(42)
    if args.sample and args.sample < len(judge_inputs):
        judge_inputs = random.sample(judge_inputs, args.sample)

    sem = asyncio.Semaphore(1)
    tasks = [judge_item(sem, item) for item in judge_inputs]
    results = await asyncio.gather(*tasks)

    with open(args.output, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved judge results to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
