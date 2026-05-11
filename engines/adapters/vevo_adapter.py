"""
VEVO Engine Adapter - Engine-specific adapter for VEVO voice conversion
Provides standardized interface for VEVO operations in the voice changer system.

VEVO (from Amphion) supports two modes:
  - "timbre": Style-preserved VC using flow-matching only (lighter)
  - "voice": Full VC using autoregressive + flow-matching
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple

import sys
import os

# Add project root to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class VevoEngineAdapter:
    """Engine-specific adapter for VEVO zero-shot voice conversion."""

    def __init__(self):
        self.engine_type = "vevo"
        self.config: Dict[str, Any] = {}
        self._engine = None

    # ------------------------------------------------------------------
    # Lazy engine creation
    # ------------------------------------------------------------------

    def _get_engine(self):
        """Return the underlying VevoEngine, creating it on first access."""
        if self._engine is None:
            from engines.vevo.vevo_engine import VevoEngine

            device = self.config.get("device")
            self._engine = VevoEngine(device=device)
        return self._engine

    # ------------------------------------------------------------------
    # Audio format helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _audio_dict_to_numpy(audio_dict: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """
        Convert a ComfyUI AUDIO dict to a mono float32 numpy array.

        Args:
            audio_dict: Dict with ``waveform`` (Tensor, shape ``(B, C, N)``) and
                        ``sample_rate`` (int).

        Returns:
            Tuple of ``(audio_numpy_1d, sample_rate)``.
        """
        waveform = audio_dict["waveform"]  # (batch, channels, samples)
        sr = audio_dict["sample_rate"]

        # Detach and move to CPU
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu()
            if waveform.dtype == torch.bfloat16:
                waveform = waveform.float()
        else:
            waveform = torch.tensor(waveform)

        # Collapse to mono 1-D
        audio_np = waveform.squeeze().numpy().astype(np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)

        return audio_np, int(sr)

    @staticmethod
    def _numpy_to_audio_dict(audio_np: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Convert mono numpy audio back to a ComfyUI AUDIO dict.

        Args:
            audio_np: 1-D float32 numpy array.
            sr: Sample rate.

        Returns:
            Dict with ``waveform`` (Tensor shape ``(1, 1, N)``) and ``sample_rate``.
        """
        tensor = torch.from_numpy(audio_np).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # (1, C, N)
        return {"waveform": tensor, "sample_rate": sr}

    # ------------------------------------------------------------------
    # Public conversion interface
    # ------------------------------------------------------------------

    def convert_voice(
        self,
        source_audio: Dict[str, Any],
        reference_audio: Dict[str, Any],
        style_reference_audio: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Perform VEVO voice conversion.

        Args:
            source_audio: ComfyUI AUDIO dict -- the audio whose content to preserve.
            reference_audio: ComfyUI AUDIO dict -- the target voice / timbre.
            style_reference_audio: Optional ComfyUI AUDIO dict used as style
                reference in ``"voice"`` mode.  If *None* the *reference_audio*
                is reused for style.
            **kwargs: Overrides forwarded to the engine (``mode``,
                ``flow_matching_steps``, etc.).

        Returns:
            Tuple of ``(converted_audio_dict, info_string)``.
        """
        engine = self._get_engine()

        # Unpack audio dicts to numpy
        src_np, src_sr = self._audio_dict_to_numpy(source_audio)
        ref_np, ref_sr = self._audio_dict_to_numpy(reference_audio)

        # Optional style reference
        style_np = None
        style_sr = None
        if style_reference_audio is not None:
            style_np, style_sr = self._audio_dict_to_numpy(style_reference_audio)

        # Merge config defaults with per-call overrides
        mode = kwargs.get("mode", self.config.get("mode", "timbre"))
        steps = kwargs.get(
            "flow_matching_steps",
            self.config.get("flow_matching_steps", 32),
        )

        print(f"[VEVO] Converting: mode={mode}, steps={steps}")

        result_np, result_sr = engine.convert_voice(
            source_audio=src_np,
            source_sr=src_sr,
            reference_audio=ref_np,
            reference_sr=ref_sr,
            mode=mode,
            flow_matching_steps=steps,
            style_reference_audio=style_np,
            style_reference_sr=style_sr,
        )

        # Build output
        out_dict = self._numpy_to_audio_dict(result_np, result_sr)
        duration = len(result_np) / result_sr
        info = (
            f"VEVO {mode} conversion | "
            f"steps={steps} | "
            f"output={result_sr}Hz | "
            f"duration={duration:.2f}s"
        )
        print(f"[VEVO] {info}")

        return out_dict, info

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Release engine resources."""
        self._engine = None
        print("[VEVO] Adapter cleanup completed.")
