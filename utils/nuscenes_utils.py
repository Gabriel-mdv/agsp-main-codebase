"""
utils/nuscenes_utils.py
───────────────────────
nuScenes safety-critical object detection for Phase 3 (H3 hypothesis).

PURPOSE
───────
H3 hypothesis: AGSP's advantage over MBP is larger for safety-critical
object categories (pedestrians, stop signs, traffic lights, cyclists,
vehicles) than for general scene understanding.

This evaluation is OPTIONAL. Run via --run_safety in phase3_eval.py.
It requires nuScenes data which is large (~300 GB for full dataset).
The mini split (~4 GB) is sufficient for this evaluation.

EVALUATION METHOD
─────────────────
We use an open-ended, non-leaky prompt to avoid answer leakage:
  "What objects are visible in this image? List all you can see."

Ground truth is determined by nuScenes annotation metadata (preferred)
or filename-based keyword matching (fallback). Prediction is positive
for a category if any keyword from that category appears in the response.

WHY OPEN-ENDED PROMPT?
──────────────────────
A leaky prompt like "Is there a pedestrian in this image? Yes or no."
gives the model the answer category, biasing the evaluation. The
open-ended prompt forces the model to generate its own object list,
which is a fairer test of visual grounding capability.

DATA LAYOUT EXPECTED
────────────────────
  data/nuscenes/
    v1.0-mini/               ← or v1.0-trainval
      scene.json
      sample.json
      sample_data.json
      category.json
      sample_annotation.json
    samples/
      CAM_FRONT/             ← front camera images (.jpg)
      CAM_FRONT_LEFT/
      CAM_FRONT_RIGHT/

SAFETY CATEGORIES
─────────────────
  pedestrian  · vehicle  · stop sign  · traffic light  · cyclist

nuScenes does not have a dedicated stop sign or traffic light class.
These are approximated using filename heuristics and scene descriptions
when metadata is unavailable.
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# nuScenes category → safety category mapping
# ─────────────────────────────────────────────────────────────────────────────

# Maps nuScenes annotation category names to our five safety categories.
# nuScenes has no dedicated stop sign or traffic light class — these fall
# back to filename keyword matching.
CATEGORY_MAP: Dict[str, List[str]] = {
    "pedestrian": [
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.wheelchair",
        "human.pedestrian.stroller",
        "human.pedestrian.personal_mobility",
    ],
    "vehicle": [
        "vehicle.car",
        "vehicle.truck",
        "vehicle.bus.rigid",
        "vehicle.bus.bendy",
        "vehicle.emergency.ambulance",
        "vehicle.emergency.police",
        "vehicle.trailer",
    ],
    "stop sign": [],     # no nuScenes class — filename heuristic only
    "traffic light": [], # no nuScenes class — filename heuristic only
    "cyclist": [
        "vehicle.bicycle",
        "vehicle.motorcycle",
    ],
}

# Keywords used for open-ended response parsing
DETECTION_KEYWORDS: Dict[str, List[str]] = {
    "pedestrian":    ["pedestrian", "person", "people", "human", "walker",
                      "man", "woman", "child", "kid"],
    "vehicle":       ["car", "vehicle", "truck", "bus", "van", "suv",
                      "automobile", "taxi"],
    "stop sign":     ["stop sign", "stop", "traffic sign", "road sign"],
    "traffic light": ["traffic light", "stoplight", "signal", "light",
                      "traffic signal"],
    "cyclist":       ["bicycle", "bike", "cyclist", "motorcycle", "rider",
                      "motorcyclist"],
}

# Preferred cameras for front-view safety evaluation
PREFERRED_CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]

# Filename keywords for fallback category assignment
FILENAME_KEYWORDS: Dict[str, List[str]] = {
    "pedestrian":    ["ped", "pedestrian", "person", "human", "walk"],
    "vehicle":       ["car", "vehicle", "truck", "bus", "van"],
    "stop sign":     ["stop", "stopsign", "sign"],
    "traffic light": ["light", "traffic", "signal"],
    "cyclist":       ["bike", "bicycle", "cyclist", "moto"],
}


# ─────────────────────────────────────────────────────────────────────────────
# NuScenesSubset dataset class
# ─────────────────────────────────────────────────────────────────────────────

class NuScenesSubset:
    """
    Loads a balanced nuScenes subset for safety-critical evaluation.

    Attempts to load from nuScenes metadata JSON files first. If metadata
    is not found, falls back to filename-based category assignment.

    Parameters
    ----------
    nuscenes_root     : root directory containing v1.0-mini/ and samples/
    safety_categories : list of safety category strings to evaluate
    n_per_category    : target number of images per category
    seed              : random seed for reproducibility
    """

    def __init__(
        self,
        nuscenes_root: str,
        safety_categories: List[str],
        n_per_category: int = 100,
        seed: int = 42,
    ):
        self.root       = Path(nuscenes_root)
        self.categories = safety_categories
        self.n_per_cat  = n_per_category
        self.seed       = seed
        self.samples: List[Dict] = []

        self._load()

    def _find_meta_dir(self) -> Optional[Path]:
        """Find the nuScenes metadata directory (v1.0-mini or v1.0-trainval)."""
        for candidate in ["v1.0-mini", "v1.0-trainval", "v1.0-test", "."]:
            p = self.root / candidate
            if (p / "sample.json").exists():
                logger.info("Found nuScenes metadata at: %s", p)
                return p
        return None

    def _load(self):
        """Load samples using metadata if available, otherwise filename scan."""
        meta_dir = self._find_meta_dir()
        if meta_dir is None:
            logger.warning(
                "nuScenes metadata not found in %s. "
                "Falling back to filename-based category assignment.",
                self.root,
            )
            self._load_from_filenames()
            return

        try:
            self._load_from_metadata(meta_dir)
        except Exception as exc:
            logger.warning(
                "Metadata loading failed (%s). "
                "Falling back to filename scan.", exc,
            )
            self._load_from_filenames()

    def _load_from_metadata(self, meta_dir: Path):
        """
        Parse nuScenes JSON metadata to build category-annotated sample list.

        For each image in the preferred cameras, determines which safety
        categories are present based on annotation metadata.
        """
        # Load core metadata files
        with open(meta_dir / "sample.json") as f:
            samples_meta = {s["token"]: s for s in json.load(f)}
        with open(meta_dir / "sample_data.json") as f:
            sample_data_list = json.load(f)
        with open(meta_dir / "category.json") as f:
            category_meta = {c["token"]: c["name"] for c in json.load(f)}

        # Find annotation file (name varies across nuScenes versions)
        ann_file = None
        for candidate in ["sample_annotation.json", "object_ann.json",
                           "instance.json"]:
            p = meta_dir / candidate
            if p.exists():
                ann_file = p
                logger.info("Using annotation file: %s", ann_file)
                break

        # Build sample_token → set of annotation categories
        sample_categories: Dict[str, set] = defaultdict(set)
        if ann_file is not None:
            with open(ann_file) as f:
                annotations = json.load(f)
            for ann in annotations:
                cat_name     = category_meta.get(ann.get("category_token", ""), "")
                sample_token = ann.get("sample_token", "")
                if cat_name and sample_token:
                    sample_categories[sample_token].add(cat_name)

        # Filter to preferred cameras and key frames only
        cam_data = [
            sd for sd in sample_data_list
            if any(cam in sd.get("filename", "") for cam in PREFERRED_CAMERAS)
            and sd.get("is_key_frame", True)
        ]

        # Build per-category image lists
        cat_images: Dict[str, List[Dict]] = {c: [] for c in self.categories}

        for sd in cam_data:
            token    = sd.get("sample_token", "")
            filename = sd.get("filename", "").lstrip("/")
            filepath = self.root / "samples" / filename

            if not filepath.exists():
                # Try just the basename under CAM_FRONT
                filepath = self.root / "samples" / "CAM_FRONT" / Path(filename).name
            if not filepath.exists():
                continue

            ann_cats = sample_categories.get(token, set())

            for safety_cat in self.categories:
                nuscenes_cats = CATEGORY_MAP.get(safety_cat, [])
                if nuscenes_cats:
                    # Has a nuScenes mapping — use annotation metadata
                    has_cat = any(nc in ann_cats for nc in nuscenes_cats)
                else:
                    # No nuScenes mapping (stop sign, traffic light) —
                    # fall back to filename keyword
                    fname_lower = filepath.stem.lower()
                    keywords    = FILENAME_KEYWORDS.get(safety_cat, [safety_cat])
                    has_cat     = any(kw in fname_lower for kw in keywords)

                cat_images[safety_cat].append({
                    "path":     str(filepath),
                    "token":    token,
                    "gt":       has_cat,
                    "category": safety_cat,
                })

        self._balance_and_finalise(cat_images)

    def _load_from_filenames(self):
        """
        Fallback: scan samples/ directory and assign ground truth labels
        based on filename keywords and directory structure.
        """
        images_dir = self.root / "samples"
        if not images_dir.exists():
            images_dir = self.root

        all_files = list(images_dir.rglob("*.jpg"))
        logger.info("Filename scan: found %d images.", len(all_files))

        cat_images: Dict[str, List[Dict]] = {c: [] for c in self.categories}
        random.seed(self.seed)
        random.shuffle(all_files)

        for fp in all_files:
            text = (fp.stem + " " + fp.parent.name).lower()
            for cat in self.categories:
                keywords = FILENAME_KEYWORDS.get(cat, [cat.split()[0]])
                gt       = any(kw in text for kw in keywords)
                cat_images[cat].append({
                    "path":     str(fp),
                    "gt":       gt,
                    "category": cat,
                })

        self._balance_and_finalise(cat_images)

    def _balance_and_finalise(self, cat_images: Dict[str, List[Dict]]):
        """
        Sample n_per_category images per category with balanced pos/neg split.
        Targets n_per_category // 2 positives and n_per_category // 2 negatives.
        """
        random.seed(self.seed)

        for cat, items in cat_images.items():
            positives = [x for x in items if x["gt"]]
            negatives = [x for x in items if not x["gt"]]

            n_pos = min(self.n_per_cat // 2, len(positives))
            n_neg = min(self.n_per_cat - n_pos, len(negatives))
            n_pos = min(self.n_per_cat - n_neg, len(positives))  # re-balance

            selected = (
                random.sample(positives, n_pos) +
                random.sample(negatives, n_neg)
            )
            random.shuffle(selected)
            self.samples.extend(selected)

        logger.info(
            "nuScenes subset: %d total items across %d categories.",
            len(self.samples), len(self.categories),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Dict]:
        return iter(self.samples)

    def get_image(self, item: Dict) -> Optional[Image.Image]:
        """Load and return a PIL image for the given sample dict."""
        try:
            return Image.open(item["path"]).convert("RGB")
        except Exception as exc:
            logger.warning("Cannot open %s: %s", item["path"], exc)
            return None

    def summary(self) -> Dict:
        """Return per-category item counts for logging."""
        per_cat = {}
        for cat in self.categories:
            items = [s for s in self.samples if s["category"] == cat]
            n_pos = sum(1 for x in items if x["gt"])
            per_cat[cat] = {
                "total":    len(items),
                "positive": n_pos,
                "negative": len(items) - n_pos,
            }
        return per_cat


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_nuscenes_safety_eval(
    model,
    tokenizer,
    image_processor,
    nuscenes_root: str,
    safety_categories: List[str],
    n_per_category: int = 100,
    device: str = "cuda",
    max_new_tokens: int = 20,
) -> Dict:
    """
    Run nuScenes safety-critical object detection evaluation (H3).

    Uses an open-ended non-leaky prompt:
      "What objects are visible in this image? List all you can see."

    Prediction is positive for a category if any detection keyword from
    that category appears in the model's free-form response.

    Parameters
    ----------
    model             : evaluated MobileVLM V2 model
    tokenizer         : AutoTokenizer
    image_processor   : CLIPImageProcessor
    nuscenes_root     : root directory of nuScenes data
    safety_categories : list of category strings
    n_per_category    : images per category (balanced pos/neg)
    device            : cuda
    max_new_tokens    : fixed at 20 for fair latency comparison

    Returns
    -------
    dict: per-category accuracy + "average" key
    """
    dataset = NuScenesSubset(
        nuscenes_root, safety_categories,
        n_per_category=n_per_category,
        seed=42,
    )
    logger.info("nuScenes dataset summary: %s", dataset.summary())

    # Open-ended, non-leaky prompt — does not reveal the category name
    PROMPT = (
        "USER: <image>\n"
        "What objects are visible in this image? List all you can see.\n"
        "ASSISTANT:"
    )

    per_cat: Dict[str, List[int]] = {c: [] for c in safety_categories}

    for item in dataset:
        image = dataset.get_image(item)
        if image is None:
            continue

        cat = item["category"]
        gt  = item["gt"]

        try:
            enc = tokenizer(PROMPT, return_tensors="pt").to(device)
            pv  = image_processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(device, dtype=torch.bfloat16)

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    images=pv,             # 'images' key for MobileLLaMA
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            response = tokenizer.decode(out[0], skip_special_tokens=True).lower()

        except Exception as exc:
            logger.warning("Inference failed for %s: %s", item["path"], exc)
            continue

        # Check if any detection keyword for this category appears in response
        keywords = DETECTION_KEYWORDS.get(cat, [cat.split()[0]])
        pred     = any(kw in response for kw in keywords)

        # Score: 1 if prediction matches ground truth, 0 otherwise
        per_cat[cat].append(int(pred == gt))

    # Compute per-category and average accuracy
    results = {
        cat: float(sum(v) / max(len(v), 1))
        for cat, v in per_cat.items()
    }
    results["average"] = float(
        sum(results[c] for c in safety_categories) / max(len(safety_categories), 1)
    )

    logger.info("nuScenes safety results: %s", results)
    return results
