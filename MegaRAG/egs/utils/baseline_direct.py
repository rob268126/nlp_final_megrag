import argparse
import asyncio
import json
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from qwen_llm import qwen_gpt_4o_mini_complete

SYSTEM_PROMPT = "Bạn là trợ lý trả lời câu hỏi dựa trên ảnh tài liệu tiếng Việt. Trả lời ngắn gọn, chính xác, bằng tiếng Việt."

PROMPT_TEMPLATE = """Dựa vào ảnh tài liệu này, hãy trả lời câu hỏi sau một cách chính xác và ngắn gọn bằng tiếng Việt:

Câu hỏi: {question}

Trả lời:"""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-items", required=True)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(args.eval_items, "r", encoding="utf-8") as f:
        eval_items = [json.loads(line) for line in f if line.strip()]

    with open(args.pages, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # resume support
    done_ids = set()
    if args.resume and pathlib.Path(args.output).exists():
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line)["index"])
                    except Exception:
                        pass

    mode = "a" if args.resume else "w"
    with open(args.output, mode, encoding="utf-8") as fout:
        for item in eval_items:
            if item["index"] in done_ids:
                continue

            page_idx = str(item.get("page_idx", ""))
            img = pages.get(page_idx, {}).get("page_image")

            try:
                ans = await qwen_gpt_4o_mini_complete(
                    PROMPT_TEMPLATE.format(question=item["question"]),
                    input_images=[img] if img else None,
                    system_prompt=SYSTEM_PROMPT,
                    max_tokens=512,
                )
                err = None
            except Exception as e:
                ans = ""
                err = str(e)

            rec = {
                "index": item["index"],
                "question": item["question"],
                "answer": ans,
                "error": err,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[baseline] done index={item['index']}")

    print(f"Saved baseline results to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
