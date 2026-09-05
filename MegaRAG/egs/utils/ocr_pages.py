import argparse
import asyncio
import json
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from qwen_llm import qwen_gpt_4o_mini_complete

SYSTEM_PROMPT = "Bạn là trợ lý OCR tài liệu tiếng Việt, chuyên trích xuất văn bản từ ảnh trang sách, đề bài, bảng biểu."

OCR_PROMPT = """Hãy trích xuất nguyên văn tiếng Việt từ ảnh tài liệu này.

Yêu cầu:
- Giữ nguyên thứ tự nội dung.
- Nếu có bảng, hãy mô tả dạng văn bản thuần.
- Không thêm bình luận, không giải thích.
- Nếu không đọc được, trả về đúng: Không đọc được văn bản.
"""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-chars", type=int, default=3000)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        pages = json.load(f)

    for page_idx, page in pages.items():
        if str(page.get("text", "")).strip():
            continue

        img_path = page.get("page_image")
        try:
            text = await qwen_gpt_4o_mini_complete(
                OCR_PROMPT,
                input_images=[img_path],
                system_prompt=SYSTEM_PROMPT,
                max_tokens=1024,
            )
        except Exception as e:
            text = f"Không đọc được văn bản. Lỗi: {e}"

        if text.startswith("ERROR:"):
            text = "Không đọc được văn bản."

        page["text"] = text[: args.max_chars]
        print(f"OCR page {page_idx}: {len(page['text'])} chars")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"Saved OCR pages to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
