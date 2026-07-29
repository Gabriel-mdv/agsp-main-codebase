# tests/conftest.py
# ─────────────────
# Ensure MobileVLM is registered with transformers AutoConfig before any
# tests run. Without this, AutoModelForCausalLM.from_pretrained() fails
# with "Unrecognized model" for mtgv/MobileVLM_V2-1.7B.
#
# pytest automatically loads conftest.py before running any test file
# in the same directory, so this registration happens exactly once.

import os
import sys

# Add MobileVLM_code to path so mobilevlm imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MobileVLM_code"))

from mobilevlm.model.mobilellama import MobileVLMConfig, MobileLlamaForCausalLM