"""
EZ-VC Engine Adapter - Standardised adapter for EZ-VC voice conversion.
Provides the ``convert_voice(source_audio, reference_audio, **kwargs)`` interface
expected by the Voice Changer node, returning ``(audio_dict, info_string)``.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple


class EZVCEngineAdapter:
    """Engine-specific adapter for EZ-VC zero-shot voice conversion."""

    def __init__(self):
        self.engine_type = "ezvc"
        self.config: Dict[str, Any] = {}
        self._engine = None

    # ------------------------------------------------------------------
    # Lazy engine creation
    # ------------------------------------------------------------------

    def _get_engine(self):
        """Return the :class:`EZVCEngine`, creating it on first access."""
        if self._engine is None:
            from engines.ezvc.ezvc_engine import EZVCEngine

            device = self.config.get("device")
            self._engine = EZVCEngine(device=device)
        return self._engine

    # ------------------------------------------------------------------
    # Audio format helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _audio_dict_to_numpy(audio_dict: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """
        Extract a 1-D float32 numpy array and sample rate from a ComfyUI
        ``AUDIO`` dict (``{waveform: Tensor, sample_rate: int}``).

        Handles bfloat16 tensors, arbitrary batch/channel dimensions, and
        ensures the result is contiguous float32.
        """
        waveform = audio_dict["waveform"]
        sr = int(audio_dict["sample_rate"])

        if isinstance(waveform, torch.Tensor):
            # bfloat16 is not supported by numpy -- cast first
            if waveform.dtype == torch.bfloat16:
                waveform = waveform.float()
            arr = waveform.detach().cpu().numpy()
        else:
            arr = np.asarray(waveform)

        # Flatten to 1-D (handles (1,1,N), (1,N), (N,) etc.)
        arr = arr.astype(np.float32).ravel()
        return arr, sr

    @staticmethod
    def _numpy_to_audio_dict(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Wrap a 1-D numpy array back into a ComfyUI ``AUDIO`` dict with
        waveform shape ``(1, 1, N)``.
        """
        audio = np.asarray(audio, dtype=np.float32).ravel()
        tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        return {"waveform": tensor, "sample_rate": sample_rate}

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
        Convert the voice in *source_audio* to match *reference_audio*.

        Parameters
        ----------
        source_audio : dict
            ComfyUI AUDIO dict ``{waveform: Tensor, sample_rate: int}``.
        reference_audio : dict
            ComfyUI AUDIO dict for the target speaker.
        **kwargs
            Override parameters (``nfe_steps``, ``speed``).

        Returns
        -------
        tuple[dict, str]
            ``(converted_audio_dict, info_string)``
        """
        engine = self._get_engine()

        # Unpack ComfyUI audio dicts
        src_np, src_sr = self._audio_dict_to_numpy(source_audio)
        ref_np, ref_sr = self._audio_dict_to_numpy(reference_audio)

        # Read parameters from config, allow kwargs to override
        nfe_steps = kwargs.get("nfe_steps", self.config.get("nfe_steps", 12))
        speed = kwargs.get("speed", self.config.get("speed", 1.0))

        print(f"[EZ-VC] Converting voice: source {len(src_np)} samples @ {src_sr} Hz, "
              f"reference {len(ref_np)} samples @ {ref_sr} Hz")
        print(f"[EZ-VC] Parameters: nfe_steps={nfe_steps}, speed={speed}")

        converted_np, out_sr = engine.convert_voice(
            source_audio=src_np,
            source_sr=src_sr,
            reference_audio=ref_np,
            reference_sr=ref_sr,
            nfe_steps=nfe_steps,
            speed=speed,
        )

        # Build result
        result_dict = self._numpy_to_audio_dict(converted_np, out_sr)
        duration = len(converted_np) / out_sr
        info = (
            f"EZ-VC conversion complete | "
            f"{duration:.1f}s @ {out_sr} Hz | "
            f"nfe_steps={nfe_steps}, speed={speed}"
        )
        print(f"[EZ-VC] {info}")
        return result_dict, info
