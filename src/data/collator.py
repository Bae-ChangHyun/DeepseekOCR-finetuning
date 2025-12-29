"""
Data Collator for DeepSeek OCR Model

DeepSeek OCR 모델 학습을 위한 DataCollator 구현
"""

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence

from src.data.model_adapter import DeepSeekOCRAdapter


@dataclass
class DeepSeekOCRDataCollator:
    """
    DeepSeek OCR 모델 학습을 위한 Data Collator

    Args:
        tokenizer: Tokenizer
        model: Model
        image_size: Size for image patches (default: 640)
        base_size: Size for global view (default: 1024)
        crop_mode: Whether to use dynamic cropping for large images
        train_on_responses_only: If True, only train on assistant responses (mask user prompts)
    """

    tokenizer: Any
    model: Any
    image_size: int = 640
    base_size: int = 1024
    crop_mode: bool = True
    image_token_id: int = 128815
    train_on_responses_only: bool = True

    def __init__(
        self,
        tokenizer,
        model,
        image_size: int = 640,
        base_size: int = 1024,
        crop_mode: bool = True,
        train_on_responses_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.image_token_id = 128815
        self.dtype = model.dtype
        self.train_on_responses_only = train_on_responses_only

        self.image_transform = DeepSeekOCRAdapter.get_image_transform(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            self.bos_id = tokenizer.bos_token_id
        else:
            self.bos_id = 0
            logger.warning(f"tokenizer has no bos_token_id, using default: {self.bos_id}")

    def deserialize_image(self, image_data) -> Image.Image:
        """Convert image data (bytes dict, PIL Image, or path) to PIL Image in RGB mode"""
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = image_data["bytes"]
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB")
        elif isinstance(image_data, (str, Path)):
            image = Image.open(image_data)
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

    def calculate_image_token_count(
        self, image: Image.Image, crop_ratio: tuple[int, int]
    ) -> int:
        """Calculate the number of tokens this image will generate"""
        num_queries = math.ceil(
            (self.image_size // self.patch_size) / self.downsample_ratio
        )
        num_queries_base = math.ceil(
            (self.base_size // self.patch_size) / self.downsample_ratio
        )

        width_crop_num, height_crop_num = crop_ratio

        if self.crop_mode:
            img_tokens = num_queries_base * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                img_tokens += (num_queries * width_crop_num + 1) * (
                    num_queries * height_crop_num
                )
        else:
            img_tokens = num_queries * num_queries + 1

        return img_tokens

    def process_image(
        self, image: Image.Image
    ) -> tuple[list, list, list, list, tuple[int, int]]:
        """
        Process a single image based on crop_mode and size thresholds

        Returns:
            Tuple of (images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio)
        """
        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        if self.crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = DeepSeekOCRAdapter.dynamic_preprocess(
                    image,
                    min_num=2,
                    max_num=9,
                    image_size=self.image_size,
                    use_thumbnail=False,
                )

            global_view = ImageOps.pad(
                image,
                (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(
                        self.image_transform(crop_img).to(self.dtype)
                    )

            num_queries = math.ceil(
                (self.image_size // self.patch_size) / self.downsample_ratio
            )
            num_queries_base = math.ceil(
                (self.base_size // self.patch_size) / self.downsample_ratio
            )

            tokenized_image = (
                [self.image_token_id] * num_queries_base + [self.image_token_id]
            ) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += (
                    [self.image_token_id] * (num_queries * width_crop_num)
                    + [self.image_token_id]
                ) * (num_queries * height_crop_num)

        else:
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])

            if self.base_size <= 640:
                resized_image = image.resize(
                    (self.base_size, self.base_size), Image.LANCZOS
                )
                images_list.append(self.image_transform(resized_image).to(self.dtype))
            else:
                global_view = ImageOps.pad(
                    image,
                    (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean),
                )
                images_list.append(self.image_transform(global_view).to(self.dtype))

            num_queries = math.ceil(
                (self.base_size // self.patch_size) / self.downsample_ratio
            )
            tokenized_image = (
                [self.image_token_id] * num_queries + [self.image_token_id]
            ) * num_queries
            tokenized_image += [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio

    def process_single_sample(self, messages: list[dict]) -> dict[str, Any]:
        """Process a single conversation into model inputs."""
        images = []
        for message in messages:
            if "images" in message and message["images"]:
                for img_data in message["images"]:
                    if img_data is not None:
                        pil_image = self.deserialize_image(img_data)
                        images.append(pil_image)

        if not images:
            raise ValueError(
                "No images found in sample. Please ensure all samples contain images."
            )

        tokenized_str = []
        images_seq_mask = []
        images_list, images_crop_list, images_spatial_crop = [], [], []

        prompt_token_count = -1
        assistant_started = False
        image_idx = 0

        tokenized_str.append(self.bos_id)
        images_seq_mask.append(False)

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "<|Assistant|>":
                if not assistant_started:
                    prompt_token_count = len(tokenized_str)
                    assistant_started = True
                content = f"{content.strip()} {self.tokenizer.eos_token}"

            text_splits = content.split("<image>")

            for i, text_sep in enumerate(text_splits):
                tokenized_sep = DeepSeekOCRAdapter.text_encode(self.tokenizer, text_sep, bos=False, eos=False)
                tokenized_str.extend(tokenized_sep)
                images_seq_mask.extend([False] * len(tokenized_sep))

                if i < len(text_splits) - 1:
                    if image_idx >= len(images):
                        raise ValueError(
                            "Data mismatch: Found '<image>' token but no corresponding image."
                        )

                    image = images[image_idx]
                    img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(
                        image
                    )

                    images_list.extend(img_list)
                    images_crop_list.extend(crop_list)
                    images_spatial_crop.extend(spatial_crop)

                    tokenized_str.extend(tok_img)
                    images_seq_mask.extend([True] * len(tok_img))

                    image_idx += 1

        if image_idx != len(images):
            raise ValueError(
                f"Data mismatch: Found {len(images)} images but only {image_idx} '<image>' tokens were used."
            )

        if not assistant_started:
            logger.warning("No assistant message found in sample. Masking all tokens.")
            prompt_token_count = len(tokenized_str)

        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)

        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros(
                (1, 3, self.base_size, self.base_size), dtype=self.dtype
            )

        return {
            "input_ids": torch.tensor(tokenized_str, dtype=torch.long),
            "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop_tensor,
            "prompt_token_count": prompt_token_count,
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate batch of samples"""
        batch_data = []

        for feature in features:
            try:
                processed = self.process_single_sample(feature["messages"])
                batch_data.append(processed)
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                continue

        if not batch_data:
            raise ValueError("No valid samples in batch")

        input_ids_list = [item["input_ids"] for item in batch_data]
        images_seq_mask_list = [item["images_seq_mask"] for item in batch_data]
        prompt_token_counts = [item["prompt_token_count"] for item in batch_data]

        input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        images_seq_mask = pad_sequence(
            images_seq_mask_list, batch_first=True, padding_value=False
        )

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[images_seq_mask] = -100

        if self.train_on_responses_only:
            for idx, prompt_count in enumerate(prompt_token_counts):
                if prompt_count > 0:
                    labels[idx, :prompt_count] = -100

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        images_batch = []
        for item in batch_data:
            images_batch.append((item["images_crop"], item["images_ori"]))

        images_spatial_crop = torch.cat(
            [item["images_spatial_crop"] for item in batch_data], dim=0
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }
