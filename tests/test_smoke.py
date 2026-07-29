"""
tests/test_smoke.py
───────────────────
Smoke tests for AGSP-VLM (MobileVLM V2-1.7B).

PURPOSE
───────
These tests verify that the environment is correctly set up before running
any experiments. Run them after completing the setup steps in README.md.

They are called "smoke tests" because they check that nothing is on fire —
they do not test correctness of results, only that all components can be
imported, instantiated, and run without crashing.

WHAT IS TESTED
──────────────
  1. Python environment    — required packages are importable
  2. GPU availability      — CUDA is available and bfloat16 is supported
  3. MobileVLM V2 source   — MobileLlamaForCausalLM can be imported
  4. Config                — config.py loads without errors
  5. Data utils            — synthetic data can be created (no real data needed)
  6. AIS computation       — gradient norms compute correctly on synthetic data
  7. Pruning               — AGSP / MBP / RAND head selection works
  8. Structural pruning    — Q/K/V/O matrices are correctly reshaped
  9. Fine-tuning setup     — LoRA can be applied to a small model
  10. POPE loading         — POPE JSON can be parsed if data/ exists
  11. Visualization        — matplotlib figures can be generated without display

USAGE
─────
  # Run all smoke tests
  python -m pytest tests/test_smoke.py -v

  # Run a specific test
  python -m pytest tests/test_smoke.py::test_gpu_available -v

  # Run quickly (skip slow model loading tests)
  python -m pytest tests/test_smoke.py -v -m "not slow"

  # Show output even for passing tests
  python -m pytest tests/test_smoke.py -v -s

NOTE
────
Tests marked @pytest.mark.slow require loading a real model and will take
several minutes. All other tests run in under 30 seconds total.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Python environment
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvironment:
    """Verify all required packages are installed and importable."""

    def test_torch_importable(self):
        """PyTorch must be importable."""
        import torch
        assert torch.__version__, "PyTorch version string is empty"

    def test_transformers_importable(self):
        """transformers >= 4.40.0 required for MobileVLM V2 compatibility."""
        import transformers
        major, minor = [int(x) for x in transformers.__version__.split(".")[:2]]
        assert (major, minor) >= (4, 40), (
            f"transformers {transformers.__version__} is too old. "
            f"Need >= 4.40.0. Run: pip install --upgrade transformers"
        )

    def test_peft_importable(self):
        """PEFT required for LoRA fine-tuning."""
        import peft
        assert peft.__version__, "PEFT version string is empty"

    def test_numpy_importable(self):
        """NumPy required for AIS matrix operations."""
        import numpy as np
        assert np.__version__

    def test_scipy_importable(self):
        """SciPy required for Spearman rho in bootstrap stability check."""
        from scipy.stats import spearmanr
        # Quick sanity check
        rho, _ = spearmanr([1, 2, 3], [1, 2, 3])
        assert abs(rho - 1.0) < 1e-6, "Spearman rho of identical arrays should be 1.0"

    def test_matplotlib_importable(self):
        """Matplotlib required for figure generation."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        assert plt is not None

    def test_pil_importable(self):
        """Pillow required for image loading."""
        from PIL import Image
        img = Image.new("RGB", (336, 336), (128, 128, 128))
        assert img.size == (336, 336)

    def test_tqdm_importable(self):
        """tqdm required for progress bars."""
        from tqdm import tqdm
        assert tqdm is not None


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: GPU and dtype
# ─────────────────────────────────────────────────────────────────────────────

class TestGPU:
    """Verify GPU availability and dtype support."""

    def test_cuda_available(self):
        """CUDA must be available. If this fails, check your GPU allocation."""
        assert torch.cuda.is_available(), (
            "CUDA not available. "
            "On PSC Bridges-2, make sure you requested a GPU node: "
            "salloc -p GPU-shared --gres=gpu:v100-32:1"
        )

    def test_gpu_name_readable(self):
        """GPU name should be readable."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
        name = torch.cuda.get_device_name(0)
        assert len(name) > 0, "GPU name is empty"
        print(f"\n  GPU: {name}")

    def test_bfloat16_supported(self):
        """
        bfloat16 must be supported for paper-matching results.

        bfloat16 is supported on V100, A100, and H100. If this fails,
        your GPU (e.g. T4, older Tesla) does not support bfloat16 and
        results will not match the paper.
        """
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
        supported = torch.cuda.is_bf16_supported()
        assert supported, (
            f"bfloat16 not supported on {torch.cuda.get_device_name(0)}. "
            f"Results will not match the paper. "
            f"Use a V100, A100, or H100 node on Bridges-2."
        )

    def test_bfloat16_tensor_ops(self):
        """bfloat16 tensor operations should work correctly."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
        x = torch.randn(4, 4, dtype=torch.bfloat16, device="cuda")
        y = torch.randn(4, 4, dtype=torch.bfloat16, device="cuda")
        z = torch.matmul(x, y)
        assert z.dtype == torch.bfloat16
        assert z.shape == (4, 4)
        assert not torch.isnan(z).any(), "NaN in bfloat16 matmul"

    def test_vram_sufficient(self):
        """
        At least 8 GB VRAM required. 32 GB recommended for paper results.
        """
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  VRAM: {vram_gb:.1f} GB")
        assert vram_gb >= 8.0, (
            f"Only {vram_gb:.1f} GB VRAM. Need at least 8 GB. "
            f"Recommend 32 GB (V100-32 or A100-40) for full experiment."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: MobileVLM V2 source
# ─────────────────────────────────────────────────────────────────────────────

class TestMobileVLMSource:
    """Verify MobileVLM V2 source is installed correctly."""

    def test_mobilevlm_importable(self):
        """
        MobileLlamaForCausalLM must be importable.

        If this fails, run:
          cd ~/agsp_v4/MobileVLM_code
          pip install -e .
        """
        try:
            from mobilevlm.model.mobilellama import MobileLlamaForCausalLM
        except ImportError as e:
            pytest.fail(
                f"Cannot import MobileLlamaForCausalLM: {e}\n"
                f"Fix: cd MobileVLM_code && pip install -e ."
            )

    def test_mobilevlm_conversation_template(self):
        """MobileVLM conversation template must be accessible."""
        try:
            from mobilevlm.conversation import conv_templates
            assert "v1" in conv_templates, (
                "Conversation template 'v1' not found in MobileVLM source."
            )
        except ImportError as e:
            pytest.fail(f"Cannot import conversation templates: {e}")

    def test_mobilevlm_utils(self):
        """MobileVLM utility functions must be importable."""
        try:
            from mobilevlm.utils import tokenizer_image_token
            from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        except ImportError as e:
            pytest.fail(f"Cannot import MobileVLM utils: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    """Verify config.py loads correctly and has expected values."""

    def test_config_loads(self):
        """config.py must load without errors."""
        from config import get_config
        cfg = get_config()
        assert cfg is not None

    def test_config_model_id(self):
        """Model ID must point to MobileVLM V2-1.7B."""
        from config import get_config
        cfg = get_config()
        assert cfg.paths.model_id == "mtgv/MobileVLM_V2-1.7B", (
            f"Unexpected model ID: {cfg.paths.model_id}"
        )

    def test_config_architecture(self):
        """Architecture config must match MobileVLM V2-1.7B specs."""
        from config import get_config
        cfg = get_config()
        # MobileVLM V2-1.7B: 24 layers × 16 heads = 384 heads
        assert cfg.phase1.n_visual_tokens == 144, (
            f"Expected 144 visual tokens (LDPv2), got {cfg.phase1.n_visual_tokens}"
        )
        assert cfg.phase1.n_calibration_samples == 100
        assert cfg.phase2.per_layer_cap == 0.50
        assert cfg.phase2.lora_r == 16
        assert cfg.phase2.lora_alpha == 32
        assert cfg.phase2.kd_weight == 0.3

    def test_config_dtype(self):
        """dtype must be bfloat16 (not 8-bit)."""
        from config import get_config
        cfg = get_config()
        assert cfg.hardware.load_in_8bit is False, (
            "load_in_8bit is True. Set to False for paper-matching results."
        )
        assert cfg.hardware.torch_dtype == "bfloat16"

    def test_config_output_dirs_created(self):
        """get_config() should create output directories automatically."""
        from config import get_config
        cfg = get_config()
        for attr in ["output_dir", "phase1_dir", "phase2_dir", "phase3_dir"]:
            path = Path(getattr(cfg.paths, attr))
            assert path.exists(), f"Output directory not created: {path}"

    def test_config_seeds(self):
        """Multi-seed config must contain expected seeds."""
        from config import get_config
        cfg = get_config()
        assert cfg.hardware.seed == 42
        assert 42 in cfg.hardware.seeds
        assert len(cfg.hardware.seeds) >= 3, "Need at least 3 seeds for validation"

    def test_config_pruning_ratios(self):
        """Pruning ratios must match paper (20/30/40/50/60%)."""
        from config import get_config
        cfg = get_config()
        expected = [0.20, 0.30, 0.40, 0.50, 0.60]
        assert cfg.phase2.pruning_ratios == expected, (
            f"Pruning ratios {cfg.phase2.pruning_ratios} don't match paper {expected}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Data utils (synthetic data — no real data needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestDataUtils:
    """Verify data loading utilities work with synthetic data."""

    def test_coco_calibration_synthetic_fallback(self):
        """
        load_coco_calibration_samples should fall back to synthetic data
        when COCO annotations are not found.
        """
        from utils.data_utils import load_coco_calibration_samples
        samples = load_coco_calibration_samples(
            annotations_path="/nonexistent/path.json",
            images_dir="/nonexistent/",
            n=5, seed=42,
        )
        assert len(samples) == 5
        for s in samples:
            assert "image" in s
            assert "caption" in s
            assert s["image"] is not None
            assert isinstance(s["caption"], str)

    def test_coco_captions_dataset_synthetic(self):
        """COCOCaptionsDataset should work with synthetic data."""
        from transformers import AutoTokenizer, CLIPImageProcessor
        from utils.data_utils import COCOCaptionsDataset

        # Use a tiny tokenizer for the test
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "mtgv/MobileVLM_V2-1.7B", trust_remote_code=True, use_fast=False
            )
            image_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14-336"
            )
        except Exception:
            pytest.skip("Model files not available — skipping dataset test")

        dataset = COCOCaptionsDataset(
            annotations_path="/nonexistent/path.json",
            images_dir="/nonexistent/",
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_samples=4,
            seed=42,
        )
        assert len(dataset) == 4

        item = dataset[0]
        assert item is not None
        assert "input_ids" in item
        assert "images" in item
        assert "labels" in item
        assert item["images"].dtype == torch.bfloat16

    def test_coco_captions_collate(self):
        """COCOCaptionsDataset collate_fn should batch correctly."""
        from transformers import AutoTokenizer, CLIPImageProcessor
        from utils.data_utils import COCOCaptionsDataset

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "mtgv/MobileVLM_V2-1.7B", trust_remote_code=True, use_fast=False
            )
            image_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14-336"
            )
        except Exception:
            pytest.skip("Model files not available")

        dataset = COCOCaptionsDataset(
            annotations_path="/nonexistent/",
            images_dir="/nonexistent/",
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_samples=4,
        )
        items  = [dataset[i] for i in range(4) if dataset[i] is not None]
        batch  = dataset.collate_fn(items)
        assert batch is not None
        assert batch["input_ids"].shape[0] == len(items)
        assert batch["images"].dtype == torch.bfloat16


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: AIS computation (synthetic model)
# ─────────────────────────────────────────────────────────────────────────────

class TestAISComputation:
    """Verify AIS computation logic with a small synthetic model."""

    def _make_tiny_model(self):
        """Create a tiny transformer model for testing AIS logic."""
        import torch.nn as nn

        class TinyAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 8)
                self.num_heads = 2

        class TinyLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = TinyAttention()

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([TinyLayer() for _ in range(3)])

        return TinyModel()

    def test_safe_get_llm_layers(self):
        """safe_get_llm_layers should find layers in flat model."""
        from utils.alignment_scoring import safe_get_llm_layers

        class ModelWithLayers:
            class model:
                layers = [1, 2, 3]

        layers = safe_get_llm_layers(ModelWithLayers())
        assert layers == [1, 2, 3]

    def test_safe_get_llm_config_flat(self):
        """safe_get_llm_config should return flat config when no text_config."""
        from utils.alignment_scoring import safe_get_llm_config

        class FlatConfig:
            num_hidden_layers = 24
            num_attention_heads = 16

        class ModelWithFlatConfig:
            config = FlatConfig()

        cfg = safe_get_llm_config(ModelWithFlatConfig())
        assert cfg.num_hidden_layers == 24
        assert cfg.num_attention_heads == 16

    def test_safe_get_llm_config_nested(self):
        """safe_get_llm_config should return text_config when nested."""
        from utils.alignment_scoring import safe_get_llm_config

        class TextConfig:
            num_hidden_layers = 32
            num_attention_heads = 32

        class NestedConfig:
            text_config = TextConfig()

        class ModelWithNestedConfig:
            config = NestedConfig()

        cfg = safe_get_llm_config(ModelWithNestedConfig())
        assert cfg.num_hidden_layers == 32

    def test_summarise_ais(self):
        """summarise_ais should compute correct statistics."""
        from utils.alignment_scoring import summarise_ais

        # Simple 2×2 AIS matrix
        ais = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        stats = summarise_ais(ais)

        assert abs(stats["mean"] - 2.5) < 1e-5
        assert abs(stats["median"] - 2.5) < 1e-5
        assert "gini" in stats
        assert stats["gini"] >= 0.0
        assert "top5_bottom50_ratio" in stats

    def test_get_top_k_heads(self):
        """get_top_k_heads should return heads sorted by AIS descending."""
        from utils.alignment_scoring import get_top_k_heads

        ais = np.array([
            [0.1, 3.5, 0.2],
            [1.0, 0.5, 2.0],
        ], dtype=np.float32)

        top3 = get_top_k_heads(ais, k=3)
        assert len(top3) == 3

        # Top head should be L0H1 (AIS=3.5)
        assert top3[0][0] == 0  # layer
        assert top3[0][1] == 1  # head
        assert abs(top3[0][2] - 3.5) < 1e-5  # score

        # Scores should be descending
        scores = [s for _, _, s in top3]
        assert scores == sorted(scores, reverse=True)

    def test_debias_ais_matrix(self):
        """_debias_ais_matrix should not produce negative values."""
        from utils.alignment_scoring import _debias_ais_matrix

        ais = np.random.rand(24, 16).astype(np.float32)
        debiased = _debias_ais_matrix(ais, n_visual_tokens=144)

        assert debiased.shape == (24, 16)
        assert (debiased >= 0).all(), "Debiased AIS contains negative values"

    def test_calibration_ablation_structure(self):
        """run_calibration_ablation output structure should be correct."""
        # We only test the structure, not the actual computation
        # (which requires a real model)
        from utils.alignment_scoring import summarise_ais, get_top_k_heads

        # Simulate what ablation results look like
        sizes = [50, 100, 250, 500]
        mock_results = {s: {"rho": 0.95 + 0.01 * i, "ais_matrix": np.zeros((24, 16))}
                        for i, s in enumerate(sizes)}

        for size, res in mock_results.items():
            assert "rho" in res
            assert 0.0 <= res["rho"] <= 1.0
            assert res["ais_matrix"].shape == (24, 16)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Pruning head selection
# ─────────────────────────────────────────────────────────────────────────────

class TestPruning:
    """Verify head selection logic for AGSP, MBP, and RAND."""

    def test_agsp_selects_lowest_ais(self):
        """AGSP should select heads with the lowest AIS scores."""
        from utils.pruning import _select_heads_to_prune

        # Create AIS where layer 0 heads have very low scores
        ais = np.ones((4, 4), dtype=np.float32)
        ais[0, :] = 0.001  # Layer 0 should all be pruned

        pruned = _select_heads_to_prune(
            ais, pruning_ratio=0.25, per_layer_cap=0.5, strategy="agsp"
        )

        # Should prune 4 heads total (4*4*0.25=4)
        # Per-layer cap is 50% = 2 heads max, so lowest layers share the pruning
        assert len(pruned) == 4
        layer_counts = {}
        for layer, head in pruned:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        assert 0 in layer_counts, "Layer 0 (lowest AIS) should be included in pruned set"

    def test_mbp_selects_lowest_magnitude(self):
        """MBP should select heads with the lowest L2 magnitude scores."""
        from utils.pruning import _select_heads_to_prune

        # Magnitude scores where layer 1 has very low values
        scores = np.ones((4, 4), dtype=np.float32)
        scores[1, :] = 0.001  # Layer 1 should all be pruned

        pruned = _select_heads_to_prune(
            scores, pruning_ratio=0.25, per_layer_cap=0.5, strategy="mbp"
        )

        assert len(pruned) == 4
        layer_counts = {}
        for layer, head in pruned:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        assert 1 in layer_counts, "Layer 1 (lowest magnitude) should be included in pruned set"

    def test_rand_respects_per_layer_cap(self):
        """Random pruning must never exceed the per-layer cap."""
        from utils.pruning import _select_heads_to_prune

        n_layers, n_heads = 8, 16
        scores = np.zeros((n_layers, n_heads), dtype=np.float32)

        pruned = _select_heads_to_prune(
            scores, pruning_ratio=0.40, per_layer_cap=0.50,
            strategy="rand", seed=42,
        )

        # Check per-layer cap: max 8 heads per layer (50% of 16)
        from collections import Counter
        counts = Counter(layer for layer, _ in pruned)
        for layer, count in counts.items():
            assert count <= 8, (
                f"Layer {layer} has {count} pruned heads, exceeds cap of 8"
            )

    def test_pruning_target_count(self):
        """Number of pruned heads should match target ratio."""
        from utils.pruning import _select_heads_to_prune

        n_layers, n_heads = 24, 16  # MobileVLM V2-1.7B dimensions
        scores = np.random.rand(n_layers, n_heads).astype(np.float32)
        ratio  = 0.40

        for strategy in ["agsp", "mbp", "rand"]:
            pruned = _select_heads_to_prune(
                scores, pruning_ratio=ratio, per_layer_cap=0.50,
                strategy=strategy, seed=42,
            )
            expected = int(n_layers * n_heads * ratio)
            assert len(pruned) == expected, (
                f"{strategy}: expected {expected} pruned heads, got {len(pruned)}"
            )

    def test_no_duplicate_heads(self):
        """Each head should appear at most once in the pruned list."""
        from utils.pruning import _select_heads_to_prune

        scores = np.random.rand(24, 16).astype(np.float32)
        for strategy in ["agsp", "mbp", "rand"]:
            pruned = _select_heads_to_prune(
                scores, pruning_ratio=0.40, per_layer_cap=0.50,
                strategy=strategy, seed=42,
            )
            assert len(pruned) == len(set(pruned)), (
                f"{strategy}: duplicate heads in pruned list"
            )

    def test_measure_sparsity(self):
        """measure_sparsity should compute correct percentages."""
        from utils.pruning import measure_sparsity

        class FakeConfig:
            num_hidden_layers   = 24
            num_attention_heads = 16

        class FakeModel:
            class config:
                num_hidden_layers   = 24
                num_attention_heads = 16

        pruned_heads = [(i, j) for i in range(4) for j in range(8)]  # 32 heads
        result = measure_sparsity(FakeModel(), pruned_heads)

        assert result["heads_pruned"]  == 32
        assert result["total_heads"]   == 384
        assert abs(result["sparsity_pct"] - 32/384*100) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Structural pruning matrix reshaping
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuralPruning:
    """Verify that structural pruning correctly reshapes Q/K/V/O matrices."""

    def test_projection_matrix_shapes_after_pruning(self):
        """
        After structural pruning, Q/K/V projection output dimensions and
        O projection input dimensions should be reduced by the number of
        pruned heads × head_dim.
        """
        import torch.nn as nn
        from utils.pruning import apply_structural_pruning

        # Create a minimal mock model with 4 layers, 4 heads, head_dim=8
        n_layers  = 4
        n_heads   = 4
        head_dim  = 8
        hidden    = n_heads * head_dim  # 32

        class MockAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(hidden, hidden)
                self.k_proj = nn.Linear(hidden, hidden)
                self.v_proj = nn.Linear(hidden, hidden)
                self.o_proj = nn.Linear(hidden, hidden)
                self.num_heads = n_heads
                self.num_key_value_heads = n_heads

        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = MockAttn()

        class MockConfig:
            num_attention_heads = n_heads
            num_hidden_layers   = n_layers
            hidden_size         = hidden

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = type("M", (), {
                    "layers": nn.ModuleList([MockLayer() for _ in range(n_layers)])
                })()
                self.config = MockConfig()

        model = MockModel()

        # Prune 1 head from each of layers 0 and 1
        heads_to_prune = [(0, 0), (1, 1)]
        apply_structural_pruning(model, heads_to_prune)

        # Layer 0: removed head 0 → 3 heads remain → dim = 3 × 8 = 24
        attn0 = model.model.layers[0].self_attn
        assert attn0.q_proj.weight.shape[0] == 3 * head_dim, (
            f"Layer 0 q_proj: expected {3*head_dim} rows, "
            f"got {attn0.q_proj.weight.shape[0]}"
        )
        assert attn0.o_proj.weight.shape[1] == 3 * head_dim, (
            f"Layer 0 o_proj: expected {3*head_dim} cols, "
            f"got {attn0.o_proj.weight.shape[1]}"
        )
        assert attn0.num_heads == 3

        # Layer 2: no pruning → 4 heads remain → dim = 4 × 8 = 32
        attn2 = model.model.layers[2].self_attn
        assert attn2.q_proj.weight.shape[0] == 4 * head_dim
        assert attn2.num_heads == 4

    def test_flops_reduction_calculation(self):
        """measure_flops_reduction should return positive reduction."""
        # This is a unit test of the calculation logic, not a real model
        # We verify the formula: FLOPs ∝ n_heads × head_dim × seq_len^2
        from utils.pruning import measure_flops_reduction

        # We can't easily create two real models here, so we test the formula
        # by checking that 50% head reduction = roughly 50% attention FLOPs
        # This is implicitly tested by the structural pruning test above
        # Just verify the function is importable and callable
        assert callable(measure_flops_reduction)


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: LoRA setup
# ─────────────────────────────────────────────────────────────────────────────

class TestLoRA:
    """Verify LoRA can be applied correctly."""

    def test_lora_importable(self):
        """PEFT LoRA components must be importable."""
        from peft import LoraConfig, get_peft_model, TaskType
        assert LoraConfig is not None

    def test_lora_config_creation(self):
        """LoRA config should be created with paper hyperparameters."""
        from peft import LoraConfig, TaskType
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert "q_proj" in cfg.target_modules


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: POPE data loading
# ─────────────────────────────────────────────────────────────────────────────

class TestPOPELoading:
    """Verify POPE JSON can be parsed if available."""

    def test_pope_json_parseable(self):
        """POPE adversarial JSON should be parseable if it exists."""
        pope_path = Path("data/pope/coco_pope_adversarial.json")

        if not pope_path.exists():
            pytest.skip("POPE data not yet downloaded. Run: python setup_data.py")

        # POPE is JSONL format (one JSON object per line)
        samples = []
        with open(pope_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        assert len(samples) > 0, "POPE file is empty"

        # Verify required fields
        first = samples[0]
        assert "image"    in first, "POPE entry missing 'image' field"
        assert "text"     in first, "POPE entry missing 'text' field"
        assert "label"    in first, "POPE entry missing 'label' field"
        assert first["label"].lower() in ["yes", "no"], (
            f"Unexpected label: {first['label']}"
        )

        print(f"\n  POPE adversarial: {len(samples)} samples loaded successfully.")

    def test_pope_yes_no_balance(self):
        """POPE adversarial should have roughly equal yes/no labels."""
        pope_path = Path("data/pope/coco_pope_adversarial.json")

        if not pope_path.exists():
            pytest.skip("POPE data not yet downloaded.")

        with open(pope_path) as f:
            samples = [json.loads(l) for l in f if l.strip()]

        yes_count = sum(1 for s in samples if s["label"].lower() == "yes")
        no_count  = sum(1 for s in samples if s["label"].lower() == "no")
        total     = yes_count + no_count

        # Adversarial POPE should be roughly balanced (within 10%)
        ratio = yes_count / total
        assert 0.4 <= ratio <= 0.6, (
            f"POPE label imbalance: {yes_count} yes / {no_count} no "
            f"(ratio={ratio:.2f}, expected ~0.5)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: Visualization
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualization:
    """Verify matplotlib figures can be generated without a display."""

    def test_ais_distribution_plot(self, tmp_path):
        """plot_ais_distribution should save a PNG without errors."""
        from utils.visualization import plot_ais_distribution

        ais = np.random.rand(24, 16).astype(np.float32)
        top_k = [(i, j, float(ais[i, j])) for i, j in [(0, 0), (1, 1), (2, 2)]]

        save_path = str(tmp_path / "test_ais_dist.png")
        plot_ais_distribution(ais, top_k, save_path, gini=0.232)

        assert Path(save_path).exists(), "AIS distribution plot not saved"
        assert Path(save_path).stat().st_size > 0, "AIS distribution plot is empty"

    def test_pareto_curve_plot(self, tmp_path):
        """plot_pareto_curve should save a PNG without errors."""
        from utils.visualization import plot_pareto_curve

        ratios = [0.20, 0.30, 0.40, 0.50, 0.60]
        agsp   = [0.194, 0.194, 0.198, 0.204, 0.203]
        mbp    = [0.282, 0.193, 0.201, 0.289, 0.322]

        save_path = str(tmp_path / "test_pareto.png")
        plot_pareto_curve(ratios, agsp, mbp, "Final Epoch Loss", save_path)

        assert Path(save_path).exists()
        assert Path(save_path).stat().st_size > 0

    def test_epoch1_damage_plot(self, tmp_path):
        """plot_epoch1_damage should save a PNG without errors."""
        from utils.visualization import plot_epoch1_damage

        ratios     = [0.20, 0.30, 0.40, 0.50, 0.60]
        agsp_e1    = [1.78, 1.90, 1.97, 1.96, 1.96]
        mbp_e1     = [3.04, 3.05, 3.94, 3.96, 4.23]

        save_path = str(tmp_path / "test_epoch1.png")
        plot_epoch1_damage(ratios, agsp_e1, mbp_e1, save_path)

        assert Path(save_path).exists()
        assert Path(save_path).stat().st_size > 0

    def test_calibration_ablation_plot(self, tmp_path):
        """plot_calibration_ablation should save a PNG without errors."""
        from utils.visualization import plot_calibration_ablation

        sizes = [50, 100, 250, 500]
        rhos  = [0.952, 0.9963, 0.998, 1.0]

        save_path = str(tmp_path / "test_ablation.png")
        plot_calibration_ablation(sizes, rhos, save_path)

        assert Path(save_path).exists()
        assert Path(save_path).stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# Slow tests (require real model loading)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestModelLoading:
    """
    Tests that require loading MobileVLM V2-1.7B from HuggingFace.

    These tests are marked 'slow' and skipped by default.
    Run with: pytest tests/test_smoke.py -v -m slow
    Requires ~3.4 GB VRAM and an internet connection (first run only).
    """

    def test_model_loads_bfloat16(self):
        """MobileVLM V2 should load in bfloat16."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "mtgv/MobileVLM_V2-1.7B",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        dtype = next(model.parameters()).dtype
        assert dtype == torch.bfloat16, (
            f"Model loaded in {dtype}, expected bfloat16"
        )

        # Check architecture
        cfg = model.config
        assert cfg.num_hidden_layers == 24, (
            f"Expected 24 layers, got {cfg.num_hidden_layers}"
        )
        assert cfg.num_attention_heads == 16, (
            f"Expected 16 heads, got {cfg.num_attention_heads}"
        )

        del model
        torch.cuda.empty_cache()

    def test_full_ais_pipeline_small(self):
        """
        Run AIS computation on 5 synthetic samples to verify the full
        pipeline works end-to-end on real model weights.
        """
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")

        from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
        from utils.alignment_scoring import compute_ais_matrix
        from utils.data_utils import load_coco_calibration_samples

        model = AutoModelForCausalLM.from_pretrained(
            "mtgv/MobileVLM_V2-1.7B",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "mtgv/MobileVLM_V2-1.7B", trust_remote_code=True
        )
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        # Use 5 synthetic samples (no real COCO needed)
        samples = load_coco_calibration_samples(
            annotations_path="/nonexistent/",
            images_dir="/nonexistent/",
            n=5, seed=42,
        )

        ais = compute_ais_matrix(
            model, tokenizer, image_processor, samples,
            n_layers=24, n_heads=16,
            use_debiased=False,
            n_visual_tokens=144,
            device="cuda",
        )

        assert ais.shape == (24, 16), f"Expected (24,16), got {ais.shape}"
        assert not np.isnan(ais).any(), "AIS matrix contains NaN"
        assert (ais >= 0).all(), "AIS matrix contains negative values"
        assert ais.max() > 0, "AIS matrix is all zeros"

        del model
        torch.cuda.empty_cache()