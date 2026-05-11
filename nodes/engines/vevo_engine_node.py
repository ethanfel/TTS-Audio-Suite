"""
VEVO Engine Node - VEVO voice conversion configuration for TTS Audio Suite
Zero-shot voice imitation using Amphion's VEVO pipeline.

Supports two modes:
  - "timbre": Style-preserved VC (flow-matching only, lighter)
  - "voice": Full VC (autoregressive + flow-matching)
"""

import os
import sys
import importlib.util
from typing import Dict, Any


# AnyType for flexible input types (accepts any data type)
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseTTSNode = base_module.BaseTTSNode


class VevoEngineNode(BaseTTSNode):
    """
    VEVO Engine configuration node - VOICE CONVERSION ONLY.

    Uses Amphion's VEVO pipeline for zero-shot voice imitation.
    Two modes available:
      - timbre: Preserves source style, converts timbre only (faster)
      - voice: Full voice conversion with AR + flow-matching (higher quality)

    This engine CANNOT generate speech from text. Use only with the
    Voice Changer node to convert audio files or microphone recordings.
    """

    @classmethod
    def NAME(cls):
        return "⚙️ VEVO Engine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["timbre", "voice"], {
                    "default": "timbre",
                    "tooltip": (
                        "Conversion mode:\n"
                        "  timbre: Style-preserved voice conversion using flow-matching only.\n"
                        "          Faster, preserves source speaking style.\n"
                        "  voice: Full voice conversion using autoregressive + flow-matching.\n"
                        "         Higher quality but slower, requires more VRAM."
                    ),
                }),
                "flow_matching_steps": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "slider",
                    "tooltip": (
                        "Number of flow-matching diffusion steps (1-64).\n"
                        "Higher = better quality but slower.\n"
                        "  8-16: Fast preview\n"
                        "  32: Recommended balance\n"
                        "  48-64: Maximum quality"
                    ),
                }),
            },
            "optional": {
                "style_reference": (any_typ, {
                    "tooltip": (
                        "Optional style reference audio for 'voice' mode.\n"
                        "Controls the speaking style independently from the timbre reference.\n"
                        "If not connected, the main reference audio is used for both timbre and style.\n"
                        "Ignored in 'timbre' mode."
                    ),
                }),
                "device": (["auto", "cuda", "xpu", "cpu", "mps"], {
                    "default": "auto",
                    "tooltip": (
                        "Processing device:\n"
                        "  auto: Automatically select best available\n"
                        "  cuda: NVIDIA GPU (requires CUDA-capable GPU)\n"
                        "  xpu: Intel GPU (requires Intel PyTorch XPU)\n"
                        "  cpu: CPU-only processing (slower)\n"
                        "  mps: Apple Metal Performance Shaders"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    DESCRIPTION = """
    VEVO Engine - Zero-Shot Voice Imitation (from Amphion)

    Voice conversion only - does NOT generate speech from text.
    Will not work with TTS Text or TTS SRT nodes. Use with Voice Changer node only.

    Two conversion modes:
      timbre: Preserves source style, changes voice timbre (faster, lighter)
      voice: Full voice conversion with AR + flow-matching (higher quality)

    Models auto-download from HuggingFace (amphion/Vevo).
    Requires git for cloning Amphion inference code on first use.
    Output sample rate: 24000 Hz.
    """

    def create_engine(
        self,
        mode="timbre",
        flow_matching_steps=32,
        style_reference=None,
        device="auto",
    ):
        """
        Create VEVO engine adapter with conversion parameters.

        Returns:
            Tuple containing VEVO engine adapter configured for voice conversion.
        """
        try:
            from engines.adapters.vevo_adapter import VevoEngineAdapter

            # Resolve device
            if device == "auto":
                import comfy.model_management as model_management
                device = str(model_management.get_torch_device())

            adapter = VevoEngineAdapter()
            adapter.config = {
                "engine_type": "vevo",
                "mode": mode,
                "flow_matching_steps": flow_matching_steps,
                "device": device,
            }

            # Store style reference on adapter if provided (voice mode)
            if style_reference is not None:
                adapter._style_reference = style_reference

            print(f"⚙️ VEVO Engine configured: mode={mode}, steps={flow_matching_steps}, device={device}")
            if style_reference is not None:
                print("   Style reference connected (will be used in 'voice' mode)")

            return (adapter,)

        except Exception as e:
            print(f"❌ VEVO Engine creation failed: {e}")
            import traceback
            traceback.print_exc()

            # Return minimal adapter on failure
            try:
                from engines.adapters.vevo_adapter import VevoEngineAdapter

                adapter = VevoEngineAdapter()
                adapter.config = {
                    "engine_type": "vevo",
                    "mode": mode,
                    "flow_matching_steps": flow_matching_steps,
                    "device": "cpu",
                    "error": str(e),
                }
                return (adapter,)
            except Exception:
                return (None,)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs for VEVO engine creation."""
        return True


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "VevoEngineNode": VevoEngineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VevoEngineNode": "⚙️ VEVO Engine",
}
