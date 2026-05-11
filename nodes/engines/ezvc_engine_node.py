"""
EZ-VC Engine Node - Zero-shot voice conversion configuration for TTS Audio Suite.
Creates an EZVCEngineAdapter with user-configurable parameters and returns it
as a TTS_ENGINE output for the Voice Changer node.
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


class EZVCEngineNode(BaseTTSNode):
    """
    EZ-VC Engine configuration node - VOICE CONVERSION ONLY.

    Uses the XEUS encoder for speech-unit extraction and an F5-TTS decoder
    with flow matching for zero-shot voice conversion.  Connect the output
    to a Voice Changer node together with source and reference audio.
    """

    @classmethod
    def NAME(cls):
        return "⚙️ EZ-VC Engine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nfe_steps": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "display": "slider",
                    "tooltip": (
                        "Number of flow-matching steps for the F5-TTS decoder. "
                        "Higher values improve quality at the cost of speed. "
                        "12 is a good default; 4-8 for fast previews, 16-32 for best quality."
                    ),
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": (
                        "Playback speed multiplier. "
                        "1.0 = natural speed, <1.0 = slower, >1.0 = faster."
                    ),
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "xpu", "cpu", "mps"], {
                    "default": "auto",
                    "tooltip": (
                        "Processing device:\n"
                        "  auto  - best available (CUDA > XPU > MPS > CPU)\n"
                        "  cuda  - NVIDIA GPU\n"
                        "  xpu   - Intel GPU\n"
                        "  mps   - Apple Silicon\n"
                        "  cpu   - CPU only (slower)"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    CATEGORY = "TTS Audio Suite/⚙️ Engines"
    FUNCTION = "create_engine"

    DESCRIPTION = (
        "EZ-VC Engine - Zero-shot Voice Conversion\n\n"
        "Voice conversion only - does NOT generate speech from text.\n"
        "Use with the Voice Changer node to convert a source voice to match "
        "a reference speaker.\n\n"
        "Based on XEUS speech-unit extraction + F5-TTS flow-matching decoder.\n"
        "Output sample rate: 16 000 Hz."
    )

    def create_engine(self, nfe_steps=12, speed=1.0, device="auto"):
        """
        Create an EZ-VC engine adapter configured with the given parameters.

        Returns
        -------
        tuple
            ``(adapter,)`` where *adapter* is an :class:`EZVCEngineAdapter`.
        """
        from engines.adapters.ezvc_adapter import EZVCEngineAdapter

        # Resolve device
        if device == "auto":
            import comfy.model_management as mm
            resolved_device = str(mm.get_torch_device())
        else:
            resolved_device = device

        adapter = EZVCEngineAdapter()
        adapter.config = {
            "engine_type": "ezvc",
            "nfe_steps": int(nfe_steps),
            "speed": float(speed),
            "device": resolved_device,
        }

        print(f"⚙️ EZ-VC Engine created | nfe_steps={nfe_steps}, speed={speed}, device={resolved_device}")
        return (adapter,)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "EZVCEngineNode": EZVCEngineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EZVCEngineNode": "⚙️ EZ-VC Engine",
}
