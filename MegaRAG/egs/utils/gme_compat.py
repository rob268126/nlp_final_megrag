from __future__ import annotations

import base64
import logging
import math
import os
from io import BytesIO
from typing import Dict, List, Optional

import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq

MODEL_NAME = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
# Giảm max pixels để an toàn hơn trên Kaggle
MAX_PIXELS = 1280 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if max(h_bar, w_bar) / min(h_bar, w_bar) > MAX_RATIO:
        logging.warning(
            f"Absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(h_bar, w_bar) / min(h_bar, w_bar)}"
        )
        if h_bar > w_bar:
            h_bar = w_bar * MAX_RATIO
        else:
            w_bar = h_bar * MAX_RATIO

    return h_bar, w_bar


def fetch_image(image, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    image_obj = None

    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image_obj = Image.open(requests.get(image, stream=True, timeout=30).raw)
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                image_obj = Image.open(BytesIO(data))
        elif os.path.exists(image):
            image_obj = Image.open(image)

    if image_obj is None:
        raise ValueError(f"Unrecognized image input: {image}")

    image = image_obj.convert("RGB")
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    image = image.resize((resized_width, resized_height))
    return image


def custom_collate_fn(batch):
    return batch


class GmeQwen2VLCompat:
    """
    wrapper tương thích transformers 4.57 cho GME-Qwen2-VL-2B-Instruct
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        min_image_tokens: int = 256,
        max_image_tokens: int = 1280,
        max_length: int = 1800,
        **kwargs,
    ) -> None:
        if device is None:
            device = os.environ.get(
                "EMBED_DEVICE",
                "cuda" if torch.cuda.is_available() else "cpu",
            )

        self.device = device

        if dtype is None:
            dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

        # Force Qwen2-VL config, tránh remote GME config
        try:
            from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
            config = Qwen2VLConfig.from_pretrained(
                model_name,
                trust_remote_code=False,
            )
        except Exception:
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=False,
            )

        config.model_type = "qwen2_vl"
        config.architectures = ["Qwen2VLForConditionalGeneration"]

        if hasattr(config, "auto_map"):
            try:
                delattr(config, "auto_map")
            except Exception:
                pass

        if not getattr(config, "_name_or_path", None):
            config._name_or_path = model_name

        self.base = AutoModelForVision2Seq.from_pretrained(
            model_name,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=False,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True,
        )

        self.base.eval()
        self.base.to(self.device)

        self.normalize = True
        self.max_length = max_length

        min_pixels = min_image_tokens * 28 * 28
        max_pixels = max_image_tokens * 28 * 28

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=False,
        )
        self.processor.tokenizer.padding_side = "right"
        self.default_instruction = "You are a helpful assistant."
        self.sep = " "

    def _get_input_embeddings(self):
        # Ưu tiên API mới: get_input_embeddings()
        if hasattr(self.base, "get_input_embeddings"):
            try:
                emb = self.base.get_input_embeddings()
                if emb is not None:
                    return emb
            except Exception:
                pass

        backbone = getattr(self.base, "model", None)

        if backbone is not None and hasattr(backbone, "get_input_embeddings"):
            try:
                emb = backbone.get_input_embeddings()
                if emb is not None:
                    return emb
            except Exception:
                pass

        # base_model.language_model.embed_tokens
        if backbone is not None and hasattr(backbone, "language_model"):
            return backbone.language_model.get_input_embeddings()

        raise AttributeError("Cannot locate input embeddings for GME wrapper.")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pooling_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        backbone = getattr(self.base, "model", self.base)

        if inputs_embeds is None:
            embed_layer = self._get_input_embeddings()
            inputs_embeds = embed_layer(input_ids)

            if pixel_values is not None:
                visual = getattr(self.base, "visual", getattr(backbone, "visual", None))
                if visual is None:
                    raise AttributeError("Cannot locate visual encoder for GME wrapper.")

                pixel_values = pixel_values.type(visual.get_dtype())
                image_embeds = visual(pixel_values, grid_thw=image_grid_thw)
                image_embeds = image_embeds.to(inputs_embeds.device)

                image_mask = input_ids == self.base.config.image_token_id
                inputs_embeds[image_mask] = image_embeds.to(inputs_embeds.dtype)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = backbone(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs.last_hidden_state
        pooling_mask = attention_mask if pooling_mask is None else pooling_mask

        left_padding = pooling_mask[:, -1].sum() == pooling_mask.shape[0]

        if left_padding:
            embeddings = hidden_states[:, -1]
        else:
            sequence_lengths = pooling_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            embeddings = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths,
            ]

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.contiguous()

    def embed(self, texts: List[str], images: List, is_query=True, instruction=None, **kwargs):
        self.base.to(self.device)
        input_texts, input_images = [], []

        for t, i in zip(texts, images):
            if not is_query or instruction is None:
                instruction = self.default_instruction

            input_str = ""

            if i is None:
                input_images = None
            else:
                input_str += "<|vision_start|><|image_pad|><|vision_end|>"
                i = fetch_image(i)
                input_images.append(i)

            if t is not None:
                input_str += t

            msg = (
                f"<|im_start|>system\n{instruction}<|im_end|>\n"
                f"<|im_start|>user\n{input_str}<|im_end|>\n"
                f"<|im_start|>assistant\n<|endoftext|>"
            )
            input_texts.append(msg)

        inputs = self.processor(
            text=input_texts,
            images=input_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            embeddings = self.forward(**inputs)

        return embeddings

    def encode(self, sentences: List[str], *, prompt_name=None, **kwargs):
        return self.get_fused_embeddings(texts=sentences, prompt_name=prompt_name, **kwargs)

    def encode_queries(self, queries: List[str], **kwargs):
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs):
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip()
                if "title" in doc
                else doc["text"].strip()
                for doc in corpus
            ]
        return self.encode(sentences, is_query=False, **kwargs)

    def get_image_embeddings(self, images, **kwargs):
        return self.get_fused_embeddings(images=images, **kwargs)

    def get_text_embeddings(self, texts: List[str], **kwargs):
        return self.get_fused_embeddings(texts=texts, **kwargs)

    def get_fused_embeddings(
        self,
        texts: Optional[List[str]] = None,
        images=None,
        **kwargs,
    ):
        if isinstance(images, DataLoader):
            image_loader = images
            batch_size = image_loader.batch_size
            if hasattr(image_loader, "dataset") and hasattr(image_loader.dataset, "transform"):
                image_loader.dataset.transform = None
        else:
            # set batch_size nhỏ 
            batch_size = kwargs.pop("batch_size", 1)

            if images is None:
                image_loader = None
            else:
                image_loader = DataLoader(
                    images,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=custom_collate_fn,
                    num_workers=0,
                )

        if texts is None:
            assert image_loader is not None
            n_batch = len(image_loader)
        else:
            n_batch = len(texts) // batch_size + int(len(texts) % batch_size > 0)
            image_loader = image_loader or [None] * n_batch

        all_embeddings = []
        none_batch = [None] * batch_size
        show_progress_bar = kwargs.pop("show_progress_bar", False)

        pbar = tqdm(
            total=n_batch,
            disable=not show_progress_bar,
            mininterval=1,
            miniters=10,
            desc="GME encode",
        )

        for n, img_batch in zip(range(0, n_batch * batch_size, batch_size), image_loader):
            text_batch = none_batch if texts is None else texts[n : n + batch_size]
            img_batch = none_batch if img_batch is None else img_batch

            embeddings = self.embed(
                texts=text_batch,
                images=img_batch,
                **kwargs,
            )

            pbar.update(1)
            all_embeddings.append(embeddings.cpu())

        pbar.close()
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings


def load_gme_model():
    """
    Hàm được gọi thay cho initialize_model() trong construct_mmkg.py / query_mmkg.py.
    """
    device = os.environ.get(
        "EMBED_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    try:
        return GmeQwen2VLCompat(device=device)
    except Exception as e:
        print(f"[gme_compat] Primary loader failed: {e}")
        print("[gme_compat] Fallback: patch require_version and load remote GME.")

        # Fallback nếu loader chính thất bại
        import transformers.utils.versions as versions
        versions.require_version = lambda *args, **kwargs: None

        from transformers import AutoModel

        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

        if not getattr(config, "_name_or_path", None):
            config._name_or_path = MODEL_NAME

        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

        model = AutoModel.from_pretrained(
            MODEL_NAME,
            config=config,
            torch_dtype=dtype,
            device_map=device if str(device).startswith("cuda") else None,
            trust_remote_code=True,
        )

        model = model.to(device)
        model.eval()
        return model
