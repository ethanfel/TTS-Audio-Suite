"""
Seed-VC Engine Node - Zero-shot voice conversion engine for TTS Audio Suite.

Provides a ComfyUI node that configures the Seed-VC voice conversion adapter
with user-selectable parameters (variant, diffusion steps, etc.).
"""

import os
import sys
import importlib.util
from typing import Dict, Any

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

from engines.adapters.seedvc_adapter import SeedVCEngineAdapter


class SeedVCEngineNode(BaseTTSNode):
    """
    Seed-VC Engine configuration node - VOICE CONVERSION ONLY.

    Configures the Seed-VC zero-shot voice conversion engine for use with
    the Voice Changer node.  This engine does NOT generate speech from text.
    """

    @classmethod
    def NAME(cls):
        return "⚙️ Seed-VC Engine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (["v2", "v1_offline", "v1_realtime"], {
                    "default": "v2",
                    "tooltip": (
                        "Seed-VC model variant:\n"
                        "  v2: Latest model with streaming support (recommended)\n"
                        "  v1_offline: Original offline model\n"
                        "  v1_realtime: Original realtime model"
                    ),
                }),
                "diffusion_steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider",
                    "tooltip": (
                        "Number of diffusion steps. Higher = better quality "
                        "but slower. 10-25 is a good range."
                    ),
                }),
                "length_adjust": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": (
                        "Output length adjustment factor. "
                        "1.0 = same length as source, <1.0 = shorter, >1.0 = longer."
                    ),
                }),
            },
            "optional": {
                "intelligibility_cfg_rate": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": (
                        "Classifier-free guidance rate for intelligibility. "
                        "Higher values preserve source content more clearly."
                    ),
                }),
                "similarity_cfg_rate": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": (
                        "Classifier-free guidance rate for speaker similarity. "
                        "Higher values make the output sound more like the reference speaker."
                    ),
                }),
                "convert_style": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Transfer speaking style (intonation, rhythm) from source. "
                        "When disabled, only the timbre is converted."
                    ),
                }),
                "device": (["auto", "cuda", "xpu", "cpu", "mps"], {
                    "default": "auto",
                    "tooltip": (
                        "Processing device:\n"
                        "  auto: Automatically select best available\n"
                        "  cuda: NVIDIA GPU\n"
                        "  xpu: Intel GPU\n"
                        "  cpu: CPU-only (slower)\n"
                        "  mps: Apple Metal (Apple Silicon)"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)

    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    FUNCTION = "create_engine"

    DESCRIPTION = (
        "Seed-VC Engine - Zero-shot Voice Conversion\n\n"
        "Voice conversion only - does NOT generate speech from text.\n"
        "Use with the Voice Changer node to convert one voice to another.\n\n"
        "Models are automatically downloaded from HuggingFace on first use.\n"
        "Supports V1 (offline/realtime) and V2 variants."
    )

    def create_engine(
        self,
        model_variant: str = "v2",
        diffusion_steps: int = 25,
        length_adjust: float = 1.0,
        intelligibility_cfg_rate: float = 0.7,
        similarity_cfg_rate: float = 0.7,
        convert_style: bool = False,
        device: str = "auto",
    ):
        """
        Create and configure a Seed-VC engine adapter.

        Returns:
            Tuple containing the configured adapter.
        """
        adapter = SeedVCEngineAdapter()

        adapter.config = {
            "engine_type": "seedvc",
            "variant": model_variant,
            "diffusion_steps": max(1, min(50, diffusion_steps)),
            "length_adjust": max(0.5, min(2.0, length_adjust)),
            "intelligibility_cfg_rate": max(0.0, min(1.0, intelligibility_cfg_rate)),
            "similarity_cfg_rate": max(0.0, min(1.0, similarity_cfg_rate)),
            "convert_style": bool(convert_style),
            "device": device,
        }

        print(
            f"⚙️ Seed-VC Engine configured: "
            f"variant={model_variant}, steps={diffusion_steps}, "
            f"length_adjust={length_adjust}, device={device}"
        )

        return (adapter,)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "SeedVCEngineNode": SeedVCEngineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVCEngineNode": "⚙️ Seed-VC Engine",
}
