import os
import asyncio
import threading
import logging
from typing import List, Optional, Any
import torch

MODEL_ID = os.environ.get(
    "QWEN_MODEL_ID",
    "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
)

_model = None
_processor = None
_load_lock = threading.Lock()
_gen_lock = threading.Lock()
logger = logging.getLogger("qwen_llm")

MAX_INPUT_CHARS = int(os.environ.get("QWEN_MAX_INPUT_CHARS", "24000"))


def _free_gpu():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _load_model():
    global _model, _processor

    with _load_lock:
        if _model is not None:
            return _model, _processor

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Qwen3-VL in this notebook.")

        n_gpu = torch.cuda.device_count()
        print(f"[qwen_llm] Detected {n_gpu} GPU(s): "
              f"{[torch.cuda.get_device_name(i) for i in range(n_gpu)]}")

        try:
            # device_map="auto" -> trải model trên tất cả GPU (2x T4 = ~32GB)
            _model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Loading with sdpa failed: {e}; fallback to default attention.")
            _model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        max_pixels = int(os.environ.get("QWEN_MAX_PIXELS", str(768 * 28 * 28)))
        try:
            _processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                min_pixels=256 * 28 * 28,
                max_pixels=max_pixels,
                trust_remote_code=True,
            )
        except TypeError:
            _processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                trust_remote_code=True
            )

        _model.eval()
        return _model, _processor


def _normalize_content(content):
    if content is None:
        return [{"type": "text", "text": ""}]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": str(content)}]


def _truncate_text(text, max_chars):
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(bị cắt bớt do quá dài)"


def _build_messages(prompt, input_images=None, system_prompt=None, history_messages=None):
    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": str(system_prompt)[:2000]}],
            }
        )

    if history_messages:
        for msg in history_messages[-4:]:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append(
                {
                    "role": role,
                    "content": _normalize_content(content),
                }
            )

    user_content = []

    if input_images:
        # tối đa 3 ảnh / lần gọi -> tránh quá tải vision tokens
        for img in input_images[:3]:
            if not img:
                continue
            if isinstance(img, str):
                if img.startswith("http://") or img.startswith("https://") or os.path.exists(img):
                    user_content.append({"type": "image", "image": img})

    user_content.append({"type": "text", "text": _truncate_text(prompt, MAX_INPUT_CHARS)})
    messages.append({"role": "user", "content": user_content})

    return messages


def _generate_once(messages, max_new_tokens):
    model, processor = _load_model()

    with torch.inference_mode():
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            k: (v.to(model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        input_len = int(inputs["input_ids"].shape[1])

        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            use_cache=True,
        )

        output_ids = generated_ids[0, input_len:]
        text = processor.batch_decode([output_ids], skip_special_tokens=True)[0]

    # Giải phóng input tensors
    del inputs, generated_ids
    _free_gpu()

    return text.strip(), input_len, int(output_ids.shape[0])


def _generate_sync(
    prompt,
    input_images=None,
    system_prompt=None,
    history_messages=None,
    max_new_tokens=1024,
):
    _load_model()

    messages = _build_messages(prompt, input_images, system_prompt, history_messages)

    with _gen_lock:
        try:
            return _generate_once(messages, max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM detected. Retrying without images and shorter prompt.")
            _free_gpu()
            messages = _build_messages(prompt, None, system_prompt, history_messages)
            try:
                return _generate_once(messages, min(max_new_tokens, 512))
            except torch.cuda.OutOfMemoryError:
                _free_gpu()
                raise


async def qwen_gpt_4o_mini_complete(
    prompt,
    input_images=None,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    token_tracker = kwargs.pop("token_tracker", None)

    kwargs.pop("response_format", None)
    kwargs.pop("hashing_kv", None)
    kwargs.pop("stream", None)

    max_new_tokens = int(kwargs.pop("max_tokens", 256 if keyword_extraction else 1024))

    try:
        text, prompt_tokens, completion_tokens = await asyncio.to_thread(
            _generate_sync,
            prompt,
            input_images,
            system_prompt,
            history_messages,
            max_new_tokens,
        )
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

    if token_tracker is not None and hasattr(token_tracker, "add_usage"):
        try:
            token_tracker.add_usage(
                {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            )
        except Exception:
            pass

    return text
