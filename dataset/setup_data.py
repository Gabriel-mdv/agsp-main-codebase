#!/usr/bin/env python3
"""
setup_data.py
─────────────
Downloads and organises all datasets required for the AGSP-VLM experiment.

PURPOSE
───────
This script is the single entry point for setting up the data directory.
Run it once before starting Phase 1. It is safe to run multiple times —
it skips files that already exist.

DATASETS DOWNLOADED
───────────────────

Required (must have before running any phase):
  1. MS-COCO val2017 images      Phase 1 calibration + Phase 2 fine-tuning
  2. MS-COCO val2017 annotations Phase 1 caption loading
  3. MS-COCO val2014 images      Phase 3 POPE evaluation
  4. POPE adversarial JSON       Phase 3 primary evaluation benchmark

Optional (needed for specific experiments):
  5. RefCOCO val JSON            Phase 2 scope boundary (Section 6.2)
  6. LLaVA-Instruct yes/no       Phase 2 Stage 2 output recalibration
  7. VQAv2 val2014               Phase 3 VQAv2 evaluation (--run_vqa flag)

Not downloaded here (too large / requires account):
  8. nuScenes                    Phase 3 H3 safety evaluation
                                 Register at https://www.nuscenes.org/
                                 and download v1.0-mini manually.

RESULTING DATA DIRECTORY STRUCTURE
───────────────────────────────────
  data/
    coco/
      annotations/
        captions_val2017.json     MS-COCO val2017 captions
      val2017/                    MS-COCO val2017 images (~5,000 images, ~1 GB)
      val2014/                    MS-COCO val2014 images (~41,000 images, ~6 GB)
    pope/
      coco_pope_adversarial.json  POPE adversarial split (500 questions)
    refcoco/
      val.json                    RefCOCO validation set
    llava_instruct/
      yesno_10324.json            10,324 balanced yes/no QA pairs
    vqa/
      v2_OpenEnded_mscoco_val2014_questions.json
      v2_mscoco_val2014_annotations.json

USAGE
─────
  # Download all required datasets (skip VQAv2)
  python setup_data.py

  # Download everything including VQAv2 (~7 GB extra)
  python setup_data.py --include_vqa

  # Skip slow image downloads (use existing images, just get annotations)
  python setup_data.py --annotations_only

  # Verify existing downloads without re-downloading
  python setup_data.py --verify_only

DISK SPACE REQUIREMENTS
───────────────────────
  Required:  ~8 GB  (COCO val2017 + val2014 + POPE + RefCOCO + LLaVA)
  With VQAv2: ~15 GB

BUGS FIXED (vs original version)
─────────────────────────────────
  Fix 1: DATA_ROOT now uses .resolve() to follow symlinks correctly
  Fix 2: extract_zip has check_dir param to avoid false-positive skip
  Fix 3: RefCOCO downloads 4 parquet files (not a single val.json)
  Fix 4: RefCOCO drops binary image bytes before JSON serialisation
  Fix 5: LLaVA yes/no extraction checks first word (not exact match)
  Fix 6: VQAv2 extract_zip uses check_dir pointing to expected JSON files
"""

import argparse
import json
import logging
import zipfile
from pathlib import Path
from typing import Dict
from urllib.request import urlretrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data directory layout
# ─────────────────────────────────────────────────────────────────────────────

# FIX 1: resolve() follows the symlink so all mkdir calls work correctly
# Without this, mkdir on a symlink target can fail with permission errors
DATA_ROOT = Path("./data").resolve()

PATHS = {
    "coco_ann_dir":    DATA_ROOT / "coco" / "annotations",
    "coco_val2017":    DATA_ROOT / "coco" / "val2017",
    "coco_val2014":    DATA_ROOT / "coco" / "val2014",
    "pope_dir":        DATA_ROOT / "pope",
    "refcoco_dir":     DATA_ROOT / "refcoco",
    "llava_dir":       DATA_ROOT / "llava_instruct",
    "vqa_dir":         DATA_ROOT / "vqa",
    "nuscenes_dir":    DATA_ROOT / "nuscenes",
}


# ─────────────────────────────────────────────────────────────────────────────
# Download utilities
# ─────────────────────────────────────────────────────────────────────────────

def _progress_hook(count: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        done = pct // 2
        bar  = "█" * done + "░" * (50 - done)
        print(f"\r  [{bar}] {pct}%", end="", flush=True)
    if count * block_size >= total_size:
        print()


def download_file(url: str, dest: Path, description: str = "") -> bool:
    if dest.exists():
        logger.info("  Already exists, skipping: %s", dest.name)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    label = description or dest.name

    logger.info("  Downloading: %s", label)
    logger.info("  URL: %s", url)
    logger.info("  Destination: %s", dest)

    try:
        urlretrieve(url, dest, reporthook=_progress_hook)
        logger.info("  Downloaded: %s  (%.1f MB)",
                    dest.name, dest.stat().st_size / 1e6)
        return True
    except Exception as exc:
        logger.error("  Download failed: %s", exc)
        if dest.exists():
            dest.unlink()
        return False


def extract_zip(
    zip_path: Path,
    dest_dir: Path,
    description: str = "",
    check_dir: Path = None,
) -> bool:
    """
    Extract a ZIP file to dest_dir.

    FIX 2: check_dir lets callers point at the specific expected output
    folder rather than the parent which may already contain other files.
    Without this fix, extract_zip would skip val2017/ extraction because
    the coco/ parent directory already existed (containing annotations/).
    """
    label = description or zip_path.name
    check = check_dir if check_dir is not None else dest_dir

    if check.exists() and any(check.iterdir()):
        logger.info("  Already extracted, skipping: %s", label)
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Extracting: %s → %s", zip_path.name, dest_dir)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            total = len(zf.namelist())
            for i, member in enumerate(zf.namelist()):
                zf.extract(member, dest_dir)
                if i % 1000 == 0:
                    print(f"\r  Extracted {i}/{total} files...", end="", flush=True)
        print()
        logger.info("  Extraction complete: %s", dest_dir)
        return True
    except Exception as exc:
        logger.error("  Extraction failed: %s", exc)
        return False


def download_from_hf(
    repo_id: str,
    filename: str,
    dest: Path,
    repo_type: str = "dataset",
) -> bool:
    if dest.exists():
        logger.info("  Already exists, skipping: %s", dest.name)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("  Downloading from HuggingFace: %s / %s", repo_id, filename)

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(dest.parent),
        )
        downloaded = Path(path)
        if downloaded != dest:
            downloaded.rename(dest)
        logger.info("  Downloaded: %s", dest.name)
        return True
    except Exception as exc:
        logger.error("  HuggingFace download failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1: MS-COCO val2017
# ─────────────────────────────────────────────────────────────────────────────

def setup_coco_val2017(annotations_only: bool = False) -> bool:
    logger.info("=" * 60)
    logger.info("Dataset 1: MS-COCO val2017 (calibration + fine-tuning)")
    logger.info("=" * 60)
    success = True

    ann_zip  = DATA_ROOT / "coco_annotations_val2017.zip"
    ann_url  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_dest = PATHS["coco_ann_dir"]

    if not (ann_dest / "captions_val2017.json").exists():
        ok = download_file(ann_url, ann_zip, "COCO 2017 annotations")
        if ok:
            extract_zip(ann_zip, DATA_ROOT / "coco", "COCO annotations")
            ann_zip.unlink(missing_ok=True)
        else:
            success = False
    else:
        logger.info("  captions_val2017.json already present.")

    if not annotations_only:
        img_zip  = DATA_ROOT / "val2017.zip"
        img_url  = "http://images.cocodataset.org/zips/val2017.zip"
        img_dest = PATHS["coco_val2017"]

        if not img_dest.exists() or not any(img_dest.iterdir()):
            ok = download_file(img_url, img_zip, "COCO val2017 images (~1 GB)")
            if ok:
                # FIX 2: check_dir points to the specific val2017/ subfolder
                extract_zip(img_zip, DATA_ROOT / "coco", "COCO val2017 images",
                            check_dir=img_dest)
                img_zip.unlink(missing_ok=True)
            else:
                success = False
        else:
            n_imgs = len(list(img_dest.glob("*.jpg")))
            logger.info("  val2017/ already present (%d images).", n_imgs)
    else:
        logger.info("  Skipping val2017 images (--annotations_only mode).")

    return success


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2: MS-COCO val2014
# ─────────────────────────────────────────────────────────────────────────────

def setup_coco_val2014(annotations_only: bool = False) -> bool:
    logger.info("=" * 60)
    logger.info("Dataset 2: MS-COCO val2014 (POPE evaluation images)")
    logger.info("=" * 60)

    img_dest = PATHS["coco_val2014"]

    if img_dest.exists() and any(img_dest.iterdir()):
        n_imgs = len(list(img_dest.glob("*.jpg")))
        logger.info("  val2014/ already present (%d images).", n_imgs)
        return True

    if annotations_only:
        logger.info("  Skipping val2014 images (--annotations_only mode).")
        return True

    img_zip = DATA_ROOT / "val2014.zip"
    img_url = "http://images.cocodataset.org/zips/val2014.zip"
    ok = download_file(img_url, img_zip, "COCO val2014 images (~6 GB, largest download)")
    if ok:
        # FIX 2: check_dir points to val2014/ specifically
        extract_zip(img_zip, DATA_ROOT / "coco", "COCO val2014 images",
                    check_dir=img_dest)
        img_zip.unlink(missing_ok=True)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 3: POPE adversarial
# ─────────────────────────────────────────────────────────────────────────────

def setup_pope() -> bool:
    logger.info("=" * 60)
    logger.info("Dataset 3: POPE adversarial evaluation")
    logger.info("=" * 60)

    pope_dir  = PATHS["pope_dir"]
    pope_dest = pope_dir / "coco_pope_adversarial.json"

    if pope_dest.exists():
        logger.info("  coco_pope_adversarial.json already present.")
        return True

    pope_dir.mkdir(parents=True, exist_ok=True)

    ok = download_from_hf(
        repo_id="lmms-lab/POPE",
        filename="coco/coco_pope_adversarial.json",
        dest=pope_dest,
        repo_type="dataset",
    )

    if not ok:
        logger.info("  Trying GitHub fallback for POPE...")
        url = (
            "https://raw.githubusercontent.com/AoiDragon/POPE/main/"
            "output/coco/coco_pope_adversarial.json"
        )
        ok = download_file(url, pope_dest, "POPE adversarial JSON")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 4: RefCOCO
# FIX 3: lmms-lab/RefCOCO is stored as 4 parquet files, not a single val.json
# FIX 4: parquet contains binary image bytes — drop before JSON serialisation
# ─────────────────────────────────────────────────────────────────────────────

def setup_refcoco() -> bool:
    logger.info("=" * 60)
    logger.info("Dataset 4: RefCOCO (scope boundary experiment, Section 6.2)")
    logger.info("=" * 60)

    refcoco_dest = PATHS["refcoco_dir"] / "val.json"

    if refcoco_dest.exists() and refcoco_dest.stat().st_size > 1000:
        logger.info("  refcoco/val.json already present.")
        return True

    PATHS["refcoco_dir"].mkdir(parents=True, exist_ok=True)

    # FIX 3: dataset is split across 4 parquet files
    parquet_files = [
        "data/val-00000-of-00004.parquet",
        "data/val-00001-of-00004.parquet",
        "data/val-00002-of-00004.parquet",
        "data/val-00003-of-00004.parquet",
    ]

    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download

        dfs = []
        for filename in parquet_files:
            logger.info("  Downloading: %s", filename)
            path = hf_hub_download(
                repo_id="lmms-lab/RefCOCO",
                filename=filename,
                repo_type="dataset",
                local_dir=str(PATHS["refcoco_dir"]),
            )
            dfs.append(pd.read_parquet(path))

        df = pd.concat(dfs, ignore_index=True)

        # FIX 4: drop binary/bytes columns that are not JSON serialisable
        # The parquet embeds raw image bytes which we don't need
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample, bytes):
                    logger.info("  Dropping binary column: %s", col)
                    df = df.drop(columns=[col])

        records = df.to_dict(orient="records")

        with open(refcoco_dest, "w") as f:
            json.dump(records, f)

        logger.info("  Saved %d records → %s", len(records), refcoco_dest)

        # Clean up parquet files to save space
        parquet_dir = PATHS["refcoco_dir"] / "data"
        if parquet_dir.exists():
            for p in parquet_dir.glob("val-*.parquet"):
                p.unlink()

        return True

    except Exception as exc:
        logger.error("  RefCOCO download failed: %s", exc)
        logger.warning(
            "  RefCOCO download failed. The scope boundary experiment "
            "(Section 6.2) will not run. Main experiment is unaffected."
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 5: LLaVA-Instruct yes/no
# FIX 5: LLaVA answers are full sentences like "Yes, the bus is red."
#         Original filter matched exact "yes"/"no" strings → found 0 pairs.
#         Fix: check the first word only, strip punctuation.
# ─────────────────────────────────────────────────────────────────────────────

def setup_llava_yesno() -> bool:
    logger.info("=" * 60)
    logger.info("Dataset 5: LLaVA-Instruct yes/no (output recalibration)")
    logger.info("=" * 60)

    llava_dest = PATHS["llava_dir"] / "yesno_10324.json"

    if llava_dest.exists() and llava_dest.stat().st_size > 1000:
        logger.info("  yesno_10324.json already present.")
        return True

    PATHS["llava_dir"].mkdir(parents=True, exist_ok=True)

    llava_src = PATHS["llava_dir"] / "llava_instruct_150k.json"

    ok = download_from_hf(
        repo_id="liuhaotian/LLaVA-Instruct-150K",
        filename="llava_instruct_150k.json",
        dest=llava_src,
        repo_type="dataset",
    )

    if ok:
        logger.info("  Extracting balanced yes/no subset (10,324 pairs)...")
        _extract_yesno_subset(src=llava_src, dest=llava_dest, target_n=10324)
    else:
        logger.warning("  LLaVA-Instruct download failed.")

    return llava_dest.exists() and llava_dest.stat().st_size > 1000


def _extract_yesno_subset(src: Path, dest: Path, target_n: int = 10324) -> None:
    logger.info("    Loading %s...", src.name)
    with open(src) as f:
        data = json.load(f)

    yes_items = []
    no_items  = []

    for item in data:
        conversations = item.get("conversations", [])
        for i, turn in enumerate(conversations):
            if (turn.get("from") == "human"
                    and i + 1 < len(conversations)
                    and conversations[i + 1].get("from") == "gpt"):

                answer = conversations[i + 1].get("value", "").strip()

                # FIX 5: check first word only — LLaVA answers are full
                # sentences like "Yes, the bus is red." not bare "yes"/"no"
                first_word = answer.split()[0].lower().rstrip(".,!?") if answer else ""

                if first_word in ("yes", "no"):
                    entry = {
                        "file_name": item.get("image", ""),
                        "caption":   turn.get("value", "").replace("<image>\n", "").strip(),
                        "answer":    first_word,
                    }
                    if first_word == "yes":
                        yes_items.append(entry)
                    else:
                        no_items.append(entry)
                    break

    logger.info("    Found %d yes and %d no items.", len(yes_items), len(no_items))

    if not yes_items and not no_items:
        logger.error("    No yes/no pairs found — check LLaVA JSON format.")
        return

    n_each = min(target_n // 2, len(yes_items), len(no_items))
    import random
    random.seed(42)
    selected = (
        random.sample(yes_items, n_each) +
        random.sample(no_items,  n_each)
    )
    random.shuffle(selected)

    with open(dest, "w") as f:
        json.dump(selected, f, indent=2)

    logger.info(
        "    Extracted %d yes/no pairs (%d yes + %d no) → %s",
        len(selected), n_each, n_each, dest.name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 6: VQAv2 (optional)
# FIX 6: extract_zip uses check_dir pointing to the expected JSON files,
#         not the vqa/ parent directory which already exists
# ─────────────────────────────────────────────────────────────────────────────

def setup_vqa() -> bool:
    logger.info("=" * 60)
    logger.info("Dataset 6: VQAv2 (optional — needed for --run_vqa)")
    logger.info("=" * 60)

    vqa_dir  = PATHS["vqa_dir"]
    q_dest   = vqa_dir / "v2_OpenEnded_mscoco_val2014_questions.json"
    ann_dest = vqa_dir / "v2_mscoco_val2014_annotations.json"

    vqa_dir.mkdir(parents=True, exist_ok=True)
    success = True

    if not q_dest.exists():
        q_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
        q_zip = vqa_dir / "v2_Questions_Val_mscoco.zip"
        ok = download_file(q_url, q_zip, "VQAv2 questions (~25 MB)")
        if ok:
            # FIX 6: check_dir points at the expected JSON, not the vqa/ dir
            extract_zip(q_zip, vqa_dir, "VQAv2 questions", check_dir=q_dest)
            q_zip.unlink(missing_ok=True)
        else:
            success = False
    else:
        logger.info("  VQAv2 questions already present.")

    if not ann_dest.exists():
        ann_url = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
        ann_zip = vqa_dir / "v2_Annotations_Val_mscoco.zip"
        ok = download_file(ann_url, ann_zip, "VQAv2 annotations (~50 MB)")
        if ok:
            # FIX 6: check_dir points at the expected JSON, not the vqa/ dir
            extract_zip(ann_zip, vqa_dir, "VQAv2 annotations", check_dir=ann_dest)
            ann_zip.unlink(missing_ok=True)
        else:
            success = False
    else:
        logger.info("  VQAv2 annotations already present.")

    return success


# ─────────────────────────────────────────────────────────────────────────────
# nuScenes instructions (manual download required)
# ─────────────────────────────────────────────────────────────────────────────

def print_nuscenes_instructions() -> None:
    logger.info("=" * 60)
    logger.info("Dataset 7: nuScenes (H3 safety evaluation — manual download)")
    logger.info("=" * 60)
    logger.info("")
    logger.info("  nuScenes requires free registration. To use H3 evaluation:")
    logger.info("")
    logger.info("  1. Register at: https://www.nuscenes.org/nuscenes")
    logger.info("  2. Download 'v1.0-mini' (4 GB) — sufficient for evaluation")
    logger.info("  3. Extract to: data/nuscenes/")
    logger.info("")
    logger.info("  Without nuScenes, Phase 3 --run_safety flag will be skipped.")
    logger.info("  The main experiment (POPE + latency) is unaffected.")
    logger.info("")


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_datasets() -> Dict[str, bool]:
    checks = {
        "COCO val2017 annotations": PATHS["coco_ann_dir"] / "captions_val2017.json",
        "COCO val2017 images":      PATHS["coco_val2017"],
        "COCO val2014 images":      PATHS["coco_val2014"],
        "POPE adversarial":         PATHS["pope_dir"] / "coco_pope_adversarial.json",
        "RefCOCO val":              PATHS["refcoco_dir"] / "val.json",
        "LLaVA yes/no":             PATHS["llava_dir"] / "yesno_10324.json",
        "VQAv2 questions":          PATHS["vqa_dir"] / "v2_OpenEnded_mscoco_val2014_questions.json",
        "VQAv2 annotations":        PATHS["vqa_dir"] / "v2_mscoco_val2014_annotations.json",
        "nuScenes mini":            PATHS["nuscenes_dir"] / "v1.0-mini" / "sample.json",
    }

    results = {}
    logger.info("")
    logger.info("=" * 60)
    logger.info("Dataset Verification Summary")
    logger.info("=" * 60)
    logger.info("  %-35s  %s", "Dataset", "Status")
    logger.info("  " + "-" * 50)

    for name, path in checks.items():
        if path.is_dir():
            exists = path.exists() and any(path.iterdir())
            count  = len(list(path.glob("*"))) if exists else 0
            status = f"✓  ({count} files)" if exists else "✗  missing"
        else:
            exists = path.exists() and path.stat().st_size > 0
            size   = f"{path.stat().st_size / 1e6:.1f} MB" if exists else ""
            status = f"✓  {size}" if exists else "✗  missing"

        results[name] = exists
        logger.info("  %-35s  %s", name, status)

    logger.info("")

    required = [
        "COCO val2017 annotations",
        "COCO val2017 images",
        "COCO val2014 images",
        "POPE adversarial",
    ]
    all_required = all(results.get(r, False) for r in required)

    if all_required:
        logger.info("  ✓ All required datasets present. Ready to run experiments.")
    else:
        missing = [r for r in required if not results.get(r, False)]
        logger.error("  ✗ Missing required datasets: %s", missing)
        logger.error("  Run: python setup_data.py")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    logger.info("AGSP-VLM Data Setup")
    logger.info("Data root: %s", DATA_ROOT.resolve())
    logger.info("")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        verify_datasets()
        return

    results = {}

    logger.info("Setting up required datasets...")
    logger.info("")

    results["coco_val2017"] = setup_coco_val2017(annotations_only=args.annotations_only)
    results["coco_val2014"] = setup_coco_val2014(annotations_only=args.annotations_only)
    results["pope"]         = setup_pope()
    results["refcoco"]      = setup_refcoco()
    results["llava"]        = setup_llava_yesno()

    if args.include_vqa:
        logger.info("")
        logger.info("Setting up optional datasets...")
        results["vqa"] = setup_vqa()

    logger.info("")
    print_nuscenes_instructions()

    verify_datasets()

    logger.info("")
    logger.info("Setup complete.")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Verify setup:  python -m pytest tests/test_smoke.py -v")
    logger.info("  2. Run Phase 1:   python src/phase1_baseline.py --n_samples 100")
    logger.info("  3. Or full run:   sbatch run_pipeline.sh")
    logger.info("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and organise datasets for AGSP-VLM")
    parser.add_argument("--include_vqa",       action="store_true")
    parser.add_argument("--annotations_only",  action="store_true")
    parser.add_argument("--verify_only",       action="store_true")
    args = parser.parse_args()
    main(args)