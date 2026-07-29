"""
utils/data_utils.py
───────────────────
Dataset loading helpers for MobileVLM V2 AGSP project.

MobileVLM V2 conversation template:
  USER: <image>\n{text}\nASSISTANT:

Datasets:
  - load_coco_calibration_samples()  → Phase 1 AIS calibration
  - COCOCaptionsDataset              → Phase 2 Stage 1 fine-tuning (paper Table 3)
  - LLaVAYesNoDataset                → Phase 2 Stage 2 recalibration
  - RefCOCODataset                   → Scope boundary experiment only (Section 6.2)

Changes from v3:
  - Added COCOCaptionsDataset — primary fine-tuning dataset matching paper.
    Phase 2 previously used RefCOCODataset which does not match Table 3.
  - Images cast to bfloat16 throughout (was float16 in v3)
  - COCOCaptionsDataset supports max_samples and seed for reproducibility
  - RefCOCODataset kept for scope boundary experiment (Section 6.2)
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data helpers (smoke tests / missing data fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_image(width: int = 336, height: int = 336) -> Image.Image:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_synthetic_samples(n: int) -> List[Dict]:
    captions = [
        "A cat sitting on a red sofa.",
        "A person walking in the park.",
        "A bicycle parked near a tree.",
        "A busy city street at night.",
        "A dog running on the beach.",
        "A plate of food on a wooden table.",
        "Children playing in a school yard.",
        "A mountain landscape at sunrise.",
        "A car driving on a highway.",
        "A bookshelf full of colourful books.",
    ]
    return [
        {"image": _make_synthetic_image(), "caption": captions[i % len(captions)]}
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: COCO calibration samples
# ─────────────────────────────────────────────────────────────────────────────

def load_coco_calibration_samples(
    annotations_path: str,
    images_dir: str,
    n: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    Load n image-caption pairs from COCO val2017 for AIS calibration.
    Falls back to synthetic data if COCO files are not found.
    """
    ann_path = Path(annotations_path)
    if not ann_path.exists():
        logger.warning(
            "COCO annotations not found at %s — using synthetic data.", ann_path
        )
        return _make_synthetic_samples(n)

    with open(ann_path) as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    random.seed(seed)
    random.shuffle(coco["annotations"])

    samples = []
    for ann in coco["annotations"]:
        if len(samples) >= n:
            break
        image_path = Path(images_dir) / id_to_file.get(ann["image_id"], "")
        if not image_path.exists():
            continue
        try:
            samples.append({
                "image":   Image.open(image_path).convert("RGB"),
                "caption": ann["caption"],
            })
        except Exception:
            pass

    if len(samples) < n:
        logger.warning(
            "Only %d / %d COCO samples loaded. Padding with synthetic data.",
            len(samples), n
        )
        samples += _make_synthetic_samples(n - len(samples))

    logger.info("Loaded %d calibration samples.", len(samples))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 Stage 1: COCO Captions fine-tuning dataset (PRIMARY — matches paper)
# ─────────────────────────────────────────────────────────────────────────────

class COCOCaptionsDataset:
    """
    COCO val2017 captions dataset for Phase 2 Stage 1 fine-tuning.

    This is the PRIMARY fine-tuning dataset matching paper Table 3:
      'Pruned models are recovered using LoRA on 6,000 COCO val2017
       image-caption pairs.'

    NOT RefCOCO — RefCOCO is used only for the scope boundary experiment
    (Section 6.2) where MBP is shown to outperform AGSP on localisation.
    """

    def __init__(
        self,
        annotations_path: str,
        images_dir: str,
        tokenizer,
        image_processor,
        max_samples: int = 6000,
        seed: int = 42,
    ):
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.images_dir      = Path(images_dir)

        ann_path = Path(annotations_path)
        if not ann_path.exists():
            logger.warning(
                "COCO annotations not found at %s — using synthetic data.", ann_path
            )
            self.data = self._synthetic_data(max_samples)
            return

        with open(ann_path) as f:
            coco = json.load(f)

        id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
        random.seed(seed)
        annotations = coco["annotations"].copy()
        random.shuffle(annotations)

        self.data = []
        for ann in annotations:
            if len(self.data) >= max_samples:
                break
            image_path = self.images_dir / id_to_file.get(ann["image_id"], "")
            if not image_path.exists():
                continue
            self.data.append({
                "file_path": str(image_path),
                "caption":   ann["caption"],
            })

        if len(self.data) < max_samples:
            logger.warning(
                "Only %d / %d COCO caption samples loaded.",
                len(self.data), max_samples
            )

        logger.info("COCOCaptionsDataset: %d items.", len(self.data))

    def _synthetic_data(self, n: int) -> List[Dict]:
        captions = [
            "A dog playing in the park.",
            "A red car on the street.",
            "Two people sitting at a table.",
            "A cat on a window sill.",
            "A bicycle leaning against a wall.",
        ]
        return [
            {"file_path": None, "caption": captions[i % len(captions)], "_synthetic": True}
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        item = self.data[idx]

        if item.get("_synthetic"):
            image = _make_synthetic_image()
        else:
            try:
                image = Image.open(item["file_path"]).convert("RGB")
            except Exception:
                return None

        caption = item["caption"]
        # MobileVLM V2 conversation template
        text = f"USER: <image>\n{caption}\nASSISTANT:"

        enc = self.tokenizer(
            text, return_tensors="pt",
            padding="max_length", max_length=256, truncation=True,
        )
        # bfloat16 to match model dtype (not float16)
        pv = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(torch.bfloat16)

        result = {k: v.squeeze(0) for k, v in enc.items()}
        result["images"] = pv.squeeze(0)
        result["labels"] = result["input_ids"].clone()
        return result

    def collate_fn(self, batch: List) -> Optional[Dict]:
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 Stage 2: LLaVA yes/no recalibration dataset
# ─────────────────────────────────────────────────────────────────────────────

class LLaVAYesNoDataset:
    """
    LLaVA-Instruct yes/no pairs for output recalibration (Stage 2 fine-tuning).

    Used to correct output bias after 40%+ compression:
      - AGSP develops yes-bias (yes-ratio 0.990) without recalibration
      - MBP develops no-bias  (yes-ratio 0.022) without recalibration

    Paper: 10,324 balanced yes/no QA pairs from LLaVA-Instruct-150K.
    LR = 5e-6, 3 epochs.
    """

    def __init__(
        self,
        json_path: str,
        images_dir: str,
        tokenizer,
        image_processor,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.images_dir      = Path(images_dir)

        with open(json_path) as f:
            raw = json.load(f)
        if max_samples:
            raw = raw[:max_samples]
        self.data = raw
        logger.info("LLaVAYesNoDataset: %d items.", len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        item = self.data[idx]
        try:
            img_path = self.images_dir / item["file_name"]
            # COCO train2014 images use prefix: COCO_train2014_000000XXXX.jpg
            # but the LLaVA JSON stores bare filenames: 000000XXXX.jpg
            if not img_path.exists():
                prefixed = f"COCO_train2014_{item['file_name']}"
                img_path = self.images_dir / prefixed
            image    = Image.open(img_path).convert("RGB")
            pv       = self.image_processor(
                images=image, return_tensors="pt"
            ).pixel_values[0].to(torch.bfloat16)

            question = item["caption"]
            answer   = item["answer"]
            prompt   = (
                f"USER: <image>\n{question} "
                f"Please answer yes or no. ASSISTANT: {answer}"
            )

            enc       = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            )
            input_ids = enc.input_ids[0]
            labels    = input_ids.clone()

            # Mask everything before ASSISTANT: — compute loss only on answer token
            assistant_ids = self.tokenizer.encode(
                "ASSISTANT:", add_special_tokens=False
            )
            for i in range(len(labels) - len(assistant_ids)):
                if labels[i:i + len(assistant_ids)].tolist() == assistant_ids:
                    labels[:i + len(assistant_ids)] = -100
                    break

            return {
                "input_ids":      input_ids,
                "attention_mask": enc.attention_mask[0],
                "images":         pv,
                "labels":         labels,
            }
        except Exception as e:
            logger.warning("LLaVAYesNo sample %d failed: %s", idx, e)
            return None

    def collate_fn(self, batch: List) -> Optional[Dict]:
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True,
            padding_value=self.tokenizer.pad_token_id or 0,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch], batch_first=True, padding_value=-100,
        )
        images = torch.stack([b["images"] for b in batch])
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "images":         images,
            "labels":         labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scope boundary experiment: RefCOCO (Section 6.2 only)
# ─────────────────────────────────────────────────────────────────────────────

class RefCOCODataset:
    """
    RefCOCO referring expressions dataset.

    Used ONLY for the scope boundary experiment (Section 6.2):
      'When Phase 2 was re-run with RefCOCO referring expressions instead
       of COCO captions, MBP achieved lower final loss (0.052 versus 0.093
       for AGSP).'

    NOT used for primary fine-tuning — use COCOCaptionsDataset instead.
    """

    def __init__(
        self,
        json_path: str,
        images_dir: str,
        tokenizer,
        image_processor,
        max_samples: Optional[int] = None,
        split: str = "train",
    ):
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.images_dir      = Path(images_dir)

        json_p = Path(json_path)
        if not json_p.exists():
            logger.warning("RefCOCO JSON not found — using synthetic data.")
            self.data = self._synthetic_data(max_samples or 200)
            return

        with open(json_p) as f:
            raw = json.load(f)
        self.data = [d for d in raw if d.get("split", split) == split]
        if max_samples:
            self.data = self.data[:max_samples]
        logger.info("RefCOCODataset [%s]: %d items.", split, len(self.data))

    def _synthetic_data(self, n: int) -> List[Dict]:
        objects = ["cat", "dog", "car", "person", "chair", "table", "laptop"]
        return [
            {
                "file_name":   f"syn_{i}.jpg",
                "expression":  f"the {objects[i % len(objects)]} on the left",
                "split":       "train",
                "_synthetic":  True,
            }
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        item = self.data[idx]
        if item.get("_synthetic"):
            image = _make_synthetic_image()
        else:
            p = self.images_dir / item.get("file_name", "")
            if not p.exists():
                return None
            try:
                image = Image.open(p).convert("RGB")
            except Exception:
                return None

        expression = item.get("expression", "Describe the main object.")
        text = f"USER: <image>\nDescribe: {expression}\nASSISTANT:"

        enc = self.tokenizer(
            text, return_tensors="pt",
            padding="max_length", max_length=256, truncation=True,
        )
        pv = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(torch.bfloat16)

        result = {k: v.squeeze(0) for k, v in enc.items()}
        result["images"] = pv.squeeze(0)
        result["labels"] = result["input_ids"].clone()
        return result

    def collate_fn(self, batch: List) -> Optional[Dict]:
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
