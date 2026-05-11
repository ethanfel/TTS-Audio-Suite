"""
Seed-VC Engine Adapter - Standardized interface for Seed-VC voice conversion.

Provides the ``convert_voice(source_audio, reference_audio, **kwargs)`` contract
expected by the Voice Changer node, returning ``(audio_dict, info_string)``.
"""

import os
import sys
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional

# Add project root for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class SeedVCEngineAdapter:
    """
    Engine adapter for Seed-VC zero-shot voice conversion.

    Converts ComfyUI AUDIO dicts (``{waveform: Tensor, sample_rate: int}``)
    to/from the numpy arrays that the underlying ``SeedVCEngine`` works with.
    """

    def __init__(self):
        self.engine_type = "seedvc"
        self.config: Dict[str, Any] = {}
        self._engine = None

    # ------------------------------------------------------------------
    # Lazy engine creation
    # ------------------------------------------------------------------

    def _get_engine(self):
        """Return the ``SeedVCEngine``, creating it on first access."""
        if self._engine is not None:
            return self._engine

        from engines.seedvc.seedvc_engine import SeedVCEngine

        device = self._resolve_device()

        self._engine = SeedVCEngine(
            variant=self.config.get("variant", "v2"),
            diffusion_steps=self.config.get("diffusion_steps", 25),
            length_adjust=self.config.get("length_adjust", 1.0),
            intelligibility_cfg_rate=self.config.get("intelligibility_cfg_rate", 0.7),
            similarity_cfg_rate=self.config.get("similarity_cfg_rate", 0.7),
            convert_style=self.config.get("convert_style", False),
            device=str(device),
        )
        return self._engine

    # ------------------------------------------------------------------
    # Audio conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _audio_dict_to_numpy(audio_dict: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """
        Extract a 1-D float32 numpy array and sample rate from a ComfyUI
        AUDIO dict (``{waveform: Tensor(B, C, N), sample_rate: int}``).

        Handles bfloat16 tensors and multi-dimensional squeezing.
        """
        waveform = audio_dict["waveform"]
        sr = int(audio_dict["sample_rate"])

        if isinstance(waveform, torch.Tensor):
            # bfloat16 cannot be directly converted to numpy
            if waveform.dtype == torch.bfloat16:
                waveform = waveform.float()
            waveform = waveform.detach().cpu().numpy()

        waveform = np.asarray(waveform, dtype=np.float32)

        # Squeeze down to 1-D: (1, 1, N) -> (N,)
        waveform = waveform.squeeze()
        if waveform.ndim > 1:
            # If still multi-dim after squeeze (e.g. stereo), average channels
            waveform = waveform.mean(axis=0)

        return waveform, sr

    @staticmethod
    def _numpy_to_audio_dict(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Wrap a 1-D float32 numpy array into a ComfyUI AUDIO dict with shape
        ``(1, 1, N)`` and peak-normalised to [-1, 1].
        """
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        # Normalise if peak > 1.0
        peak = np.abs(audio).max()
        if peak > 1.0:
            audio = audio / peak

        # Reshape to (1, 1, N)
        tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)

        return {
            "waveform": tensor,
            "sample_rate": sample_rate,
        }

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        """Resolve device from config, falling back to ComfyUI default."""
        device = self.config.get("device", "auto")
        if device is None or device == "auto":
            import comfy.model_management as model_management
            return str(model_management.get_torch_device())
        return device

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def convert_voice(
        self,
        source_audio: Dict[str, Any],
        reference_audio: Dict[str, Any],
        **kwargs,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Convert the voice of *source_audio* to match *reference_audio*.

        Both inputs are ComfyUI AUDIO dicts
        (``{waveform: Tensor(1,1,N), sample_rate: int}``).

        Returns:
            Tuple of ``(converted_audio_dict, info_string)``.
        """
        engine = self._get_engine()

        # Unpack audio dicts to numpy
        source_np, source_sr = self._audio_dict_to_numpy(source_audio)
        ref_np, ref_sr = self._audio_dict_to_numpy(reference_audio)

        variant = self.config.get("variant", "v2")
        steps = self.config.get("diffusion_steps", 25)

        print(f"Seed-VC: converting voice (variant={variant}, steps={steps}, "
              f"source_sr={source_sr}, ref_sr={ref_sr})")

        # Run conversion
        converted_np, out_sr = engine.convert_voice(
            source_audio=source_np,
            source_sr=source_sr,
            reference_audio=ref_np,
            reference_sr=ref_sr,
        )

        # Build output audio dict
        out_dict = self._numpy_to_audio_dict(converted_np, out_sr)

        # Build info string
        duration = len(converted_np) / out_sr if out_sr > 0 else 0.0
        info = (
            f"Seed-VC ({variant}) | steps={steps} | "
            f"length_adjust={self.config.get('length_adjust', 1.0)} | "
            f"output={out_sr}Hz | duration={duration:.2f}s"
        )

        return out_dict, info

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Release engine resources."""
        if self._engine is not None:
            self._engine.cleanup()
            self._engine = None
        print("Seed-VC adapter cleanup completed")
