"""
utils/vqa_utils.py
──────────────────
VQAv2 dataset loader and evaluator for Phase 3 (optional evaluation).

PURPOSE
───────
VQAv2 evaluation measures general visual question answering capability
after pruning. It answers the question: does AGSP preserve the model's
general VQA ability, or does pruning hurt everything equally?

This is a secondary metric — the primary metric is POPE adversarial.
VQAv2 is run via the --run_vqa flag in phase3_eval.py.

EVALUATION METHOD
─────────────────
VQA soft accuracy: a prediction scores 1.0 if it exactly matches any
of the 10 annotator answers (after normalisation), and partial credit
min(count/3, 1.0) if it matches some but not all annotators.

This is the official VQA evaluation metric. It rewards answers that
are partially correct (e.g. "2" instead of "two").

DATA SOURCES
────────────
1. Local COCO-format JSON files (fastest, no network required):
     data/vqa/v2_OpenEnded_mscoco_val2014_questions.json
     data/vqa/v2_mscoco_val2014_annotations.json
     data/coco/val2014/  (images)

2. HuggingFace datasets (fallback, requires internet):
     lmms-lab/VQAv2

3. Synthetic samples (smoke test fallback, no real data needed)

MobileVLM V2 prompt format:
  USER: <image>\n{question}\nASSISTANT:
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Answer normalisation (official VQA evaluation rules)
# ─────────────────────────────────────────────────────────────────────────────

# Contraction expansions from official VQA eval script
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't",
    "couldve": "could've", "couldnt": "couldn't", "didnt": "didn't",
    "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hasnt": "hasn't", "isnt": "isn't", "shouldve": "should've",
    "shouldnt": "shouldn't", "wasnt": "wasn't", "werent": "weren't",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
}

_PUNCT    = r'[!"#$%&()*+,-./:;<=>?@\[\]\\^_`{|}~]'
_ARTICLES = {"a", "an", "the"}


def _normalise_answer(s: str) -> str:
    """
    Apply official VQA answer normalisation:
    lowercase → expand contractions → remove punctuation → remove articles.
    """
    s = s.lower().strip()
    for k, v in _CONTRACTIONS.items():
        s = s.replace(k, v)
    s = re.sub(_PUNCT, " ", s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens).strip()


def _vqa_soft_accuracy(prediction: str, gt_answers: List[str]) -> float:
    """
    Official VQA soft accuracy: min(count_of_matching_annotators / 3, 1.0).

    gt_answers: list of up to 10 annotator answers.
    Returns a score in [0.0, 1.0].
    """
    pred_norm = _normalise_answer(prediction)
    count = sum(1 for a in gt_answers if _normalise_answer(a) == pred_norm)
    return min(count / 3.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_vqa_from_local(
    questions_json: str,
    annotations_json: str,
    images_dir: str,
    n: int = 2000,
    seed: int = 42,
) -> List[Dict]:
    """
    Load VQAv2 from local COCO-format JSON files.

    Expects:
      questions_json   : v2_OpenEnded_mscoco_val2014_questions.json
      annotations_json : v2_mscoco_val2014_annotations.json
      images_dir       : directory containing COCO val2014 JPEG images

    Returns list of dicts: {image, question, answers, question_id}
    """
    q_path = Path(questions_json)
    a_path = Path(annotations_json)

    if not q_path.exists() or not a_path.exists():
        logger.info("VQAv2 local files not found — will try HuggingFace.")
        return []

    with open(q_path) as f:
        q_data = json.load(f)
    with open(a_path) as f:
        a_data = json.load(f)

    # Build answer lookup: question_id → list of annotator answers
    ann_lookup: Dict[int, List[str]] = {}
    for ann in a_data["annotations"]:
        ann_lookup[ann["question_id"]] = [a["answer"] for a in ann["answers"]]

    random.seed(seed)
    questions = random.sample(
        q_data["questions"], min(n, len(q_data["questions"]))
    )

    samples = []
    for q in questions:
        qid      = q["question_id"]
        image_id = q["image_id"]
        img_name = f"COCO_val2014_{image_id:012d}.jpg"
        img_path = Path(images_dir) / img_name

        if not img_path.exists():
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        samples.append({
            "image":       image,
            "question":    q["question"],
            "answers":     ann_lookup.get(qid, []),
            "question_id": qid,
        })

    logger.info("Loaded %d VQAv2 samples from local files.", len(samples))
    return samples


def _load_vqa_from_hf(n: int = 2000, seed: int = 42) -> List[Dict]:
    """
    Load VQAv2 from HuggingFace datasets as a fallback.
    Requires internet access and the datasets package.
    """
    try:
        from datasets import load_dataset
        logger.info("Loading VQAv2 from HuggingFace (n=%d)...", n)

        ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
        random.seed(seed)
        samples = []

        for item in ds:
            if len(samples) >= n:
                break
            try:
                samples.append({
                    "image":       item["image"],
                    "question":    item["question"],
                    "answers":     [a["answer"] for a in item.get("answers", [])],
                    "question_id": item.get("question_id", len(samples)),
                })
            except Exception:
                continue

        logger.info("Loaded %d VQAv2 samples from HuggingFace.", len(samples))
        return samples

    except Exception as exc:
        logger.warning("HuggingFace VQAv2 load failed: %s", exc)
        return []


def _make_synthetic_vqa(n: int) -> List[Dict]:
    """
    Synthetic VQAv2 samples for smoke tests when no real data is available.
    Not for paper results.
    """
    import numpy as np
    questions = [
        ("What color is the sky?",     ["blue", "blue", "blue", "light blue", "blue"]),
        ("How many people are there?",  ["2", "two", "2", "2", "2"]),
        ("Is it daytime?",              ["yes", "yes", "yes", "yes", "no"]),
        ("What animal is shown?",       ["dog", "dog", "dog", "puppy", "dog"]),
        ("What sport is being played?", ["tennis", "tennis", "tennis", "tennis", "tennis"]),
    ]
    samples = []
    for i in range(n):
        q, answers = questions[i % len(questions)]
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        samples.append({
            "image":       Image.fromarray(arr),
            "question":    q,
            "answers":     answers,
            "question_id": i,
        })
    logger.warning("Using %d synthetic VQAv2 samples — NOT for paper results.", n)
    return samples


def load_vqa_samples(cfg, n: int = 2000, seed: int = 42) -> List[Dict]:
    """
    Load VQAv2 samples with automatic fallback:
      1. Local COCO-format JSON files (fastest)
      2. HuggingFace datasets (requires internet)
      3. Synthetic samples (smoke test only)

    Parameters
    ----------
    cfg  : config object (uses cfg.paths.vqa_* and cfg.paths.vqa_images_dir)
    n    : number of samples to load
    seed : random seed for reproducibility
    """
    # Try local files first
    q_json  = getattr(cfg.paths, "vqa_questions_json",   "")
    a_json  = getattr(cfg.paths, "vqa_annotations_json", "")
    img_dir = getattr(cfg.paths, "vqa_images_dir", cfg.paths.coco_val_images_dir)

    if q_json and Path(q_json).exists():
        samples = _load_vqa_from_local(q_json, a_json, img_dir, n, seed)
        if samples:
            return samples

    # Try HuggingFace
    samples = _load_vqa_from_hf(n, seed)
    if samples:
        return samples

    # Synthetic fallback
    logger.warning("No VQAv2 data found. Using synthetic fallback for smoke test.")
    return _make_synthetic_vqa(n)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_vqa_evaluation(
    model,
    tokenizer,
    image_processor,
    samples: List[Dict],
    device: str = "cuda",
    max_new_tokens: int = 20,
) -> Dict:
    """
    Run VQAv2 evaluation with soft accuracy scoring.

    MobileVLM V2 prompt format:
      USER: <image>\n{question}\nASSISTANT:

    Prediction is extracted from the ASSISTANT: portion of the output.
    Both soft accuracy (official VQA metric) and exact accuracy are reported.

    Parameters
    ----------
    model           : evaluated MobileVLM V2 model
    tokenizer       : AutoTokenizer
    image_processor : CLIPImageProcessor
    samples         : list of dicts from load_vqa_samples()
    device          : cuda
    max_new_tokens  : fixed at 20 for fair latency comparison

    Returns
    -------
    dict with keys: soft_accuracy, exact_accuracy, n_evaluated
    """
    model.eval()

    total_soft  = 0.0
    total_exact = 0
    n_evaluated = 0

    for idx, sample in enumerate(samples):
        if idx % 200 == 0:
            logger.info("VQA eval: %d / %d", idx, len(samples))

        question = sample["question"]
        image    = sample["image"]
        answers  = sample.get("answers", [])

        # MobileVLM V2 conversation template
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        try:
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            pv  = image_processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(device, dtype=torch.bfloat16)

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    images=pv,               # 'images' key for MobileLLaMA
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            raw  = tokenizer.decode(out[0], skip_special_tokens=True)
            # Extract answer from after ASSISTANT:
            pred = raw.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in raw else raw.strip()

            soft  = _vqa_soft_accuracy(pred, answers)
            exact = int(
                _normalise_answer(pred) in [_normalise_answer(a) for a in answers]
            )

            total_soft  += soft
            total_exact += exact
            n_evaluated += 1

        except Exception as exc:
            logger.warning("VQA sample %d failed: %s", idx, exc)

    return {
        "soft_accuracy":  total_soft  / max(n_evaluated, 1),
        "exact_accuracy": total_exact / max(n_evaluated, 1),
        "n_evaluated":    n_evaluated,
    }
