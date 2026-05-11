"""
VEVO Voice Conversion Engine

Zero-shot voice imitation using Amphion's VEVO pipeline.
Supports two modes:
  - "timbre": Style-preserved voice conversion (flow-matching only, lighter)
  - "voice": Full voice conversion (autoregressive + flow-matching)

References:
  - Repository: https://github.com/open-mmlab/Amphion
  - Models: https://huggingface.co/amphion/Vevo
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import torch
import torchaudio
from typing import Tuple, Optional

import folder_paths
import comfy.model_management


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMPHION_REPO_URL = "https://github.com/open-mmlab/Amphion.git"
OUTPUT_SAMPLE_RATE = 24000

# HuggingFace repo holding all VEVO model components
HF_REPO_ID = "amphion/Vevo"

# Selective download patterns for each pipeline stage
_TIMBRE_PATTERNS = [
    "tokenizer/vq8192/*",
    "Vq8192ToMels/*",
    "Vocoder/*",
]

_VOICE_PATTERNS = _TIMBRE_PATTERNS + [
    "tokenizer/vq32/*",
    "Vq32ToVq8192/*",
]

# Amphion config paths (relative to Amphion repo root)
_CFG_FM = "models/vc/vevo/config/Vq8192ToMels.json"
_CFG_VOCODER = "models/vc/vevo/config/Vocoder.json"
_CFG_AR = "models/vc/vevo/config/Vq32ToVq8192.json"


class VevoEngine:
    """
    VEVO voice conversion engine.

    Lazy-loads Amphion repo and model weights on first use.
    Provides two conversion modes:
      - timbre: preserves source style, converts timbre only (flow-matching)
      - voice: full voice conversion (AR token prediction + flow-matching)
    """

    def __init__(self, device: Optional[str] = None):
        self._device = device or str(comfy.model_management.get_torch_device())
        self._amphion_dir: Optional[str] = None
        self._model_dir: Optional[str] = None
        self._pipeline_timbre = None
        self._pipeline_voice = None

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device: str) -> "VevoEngine":
        """Move engine to *device* (lazy -- actual model move happens on next inference)."""
        self._device = device
        # Invalidate cached pipelines so they reload on the correct device
        self._pipeline_timbre = None
        self._pipeline_voice = None
        return self

    # ------------------------------------------------------------------
    # Repo / weight helpers
    # ------------------------------------------------------------------

    def _ensure_amphion(self) -> str:
        """Clone Amphion repo (depth 1) if not already present. Returns repo path."""
        if self._amphion_dir and os.path.isdir(self._amphion_dir):
            return self._amphion_dir

        base = os.path.join(folder_paths.models_dir, "TTS", "VEVO")
        os.makedirs(base, exist_ok=True)
        repo_dir = os.path.join(base, "Amphion")

        if not os.path.isdir(os.path.join(repo_dir, ".git")):
            print(f"[VEVO] Cloning Amphion repository to {repo_dir} ...")
            subprocess.check_call(
                ["git", "clone", "--depth", "1", AMPHION_REPO_URL, repo_dir],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            print("[VEVO] Amphion clone complete.")
        else:
            print(f"[VEVO] Amphion repo already present at {repo_dir}")

        self._amphion_dir = repo_dir
        return repo_dir

    def _ensure_models(self, patterns: list) -> str:
        """Download model components from HuggingFace using *patterns*. Returns local dir."""
        if self._model_dir and os.path.isdir(self._model_dir):
            # Check whether the requested patterns are already satisfied
            return self._model_dir

        from huggingface_hub import snapshot_download

        base = os.path.join(folder_paths.models_dir, "TTS", "VEVO")
        os.makedirs(base, exist_ok=True)

        print(f"[VEVO] Downloading model components (patterns={patterns}) ...")
        local_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns=patterns,
            local_dir=os.path.join(base, "models"),
        )
        print(f"[VEVO] Models ready at {local_dir}")
        self._model_dir = local_dir
        return local_dir

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _inject_amphion_path(self):
        """Ensure Amphion source is importable."""
        amphion_dir = self._ensure_amphion()
        if amphion_dir not in sys.path:
            sys.path.insert(0, amphion_dir)

    def _load_timbre_pipeline(self):
        """Lazily build the timbre (flow-matching only) pipeline."""
        if self._pipeline_timbre is not None:
            return self._pipeline_timbre

        self._inject_amphion_path()
        model_dir = self._ensure_models(_TIMBRE_PATTERNS)

        from models.vc.vevo.vevo_utils import VevoInferencePipeline

        amphion_dir = self._amphion_dir

        pipeline = VevoInferencePipeline(
            content_tokenizer_ckpt_path=os.path.join(model_dir, "tokenizer", "vq8192"),
            content_tokenizer_config_path=os.path.join(amphion_dir, _CFG_FM),
            flow_matching_ckpt_path=os.path.join(model_dir, "Vq8192ToMels"),
            flow_matching_config_path=os.path.join(amphion_dir, _CFG_FM),
            vocoder_ckpt_path=os.path.join(model_dir, "Vocoder"),
            vocoder_config_path=os.path.join(amphion_dir, _CFG_VOCODER),
            device=self._device,
        )
        self._pipeline_timbre = pipeline
        print("[VEVO] Timbre pipeline loaded.")
        return pipeline

    def _load_voice_pipeline(self):
        """Lazily build the full voice (AR + FM) pipeline."""
        if self._pipeline_voice is not None:
            return self._pipeline_voice

        self._inject_amphion_path()
        model_dir = self._ensure_models(_VOICE_PATTERNS)

        from models.vc.vevo.vevo_utils import VevoInferencePipeline

        amphion_dir = self._amphion_dir

        pipeline = VevoInferencePipeline(
            content_tokenizer_ckpt_path=os.path.join(model_dir, "tokenizer", "vq32"),
            content_tokenizer_config_path=os.path.join(amphion_dir, _CFG_AR),
            ar_ckpt_path=os.path.join(model_dir, "Vq32ToVq8192"),
            ar_config_path=os.path.join(amphion_dir, _CFG_AR),
            flow_matching_ckpt_path=os.path.join(model_dir, "Vq8192ToMels"),
            flow_matching_config_path=os.path.join(amphion_dir, _CFG_FM),
            vocoder_ckpt_path=os.path.join(model_dir, "Vocoder"),
            vocoder_config_path=os.path.join(amphion_dir, _CFG_VOCODER),
            device=self._device,
        )
        self._pipeline_voice = pipeline
        print("[VEVO] Voice (AR + FM) pipeline loaded.")
        return pipeline

    # ------------------------------------------------------------------
    # Low-level conversion helpers
    # ------------------------------------------------------------------

    def _convert_timbre(
        self,
        src_path: str,
        ref_path: str,
        steps: int = 32,
    ) -> np.ndarray:
        """Run timbre-only conversion. Returns numpy float32 mono audio."""
        pipeline = self._load_timbre_pipeline()
        gen_audio = pipeline.inference_fm(
            src_wav_path=src_path,
            timbre_ref_wav_path=ref_path,
            flow_matching_steps=steps,
        )
        return gen_audio.cpu().numpy().astype(np.float32)

    def _convert_voice(
        self,
        src_path: str,
        ref_path: str,
        style_path: str,
        steps: int = 32,
    ) -> np.ndarray:
        """Run full voice conversion (AR + FM). Returns numpy float32 mono audio."""
        pipeline = self._load_voice_pipeline()
        gen_audio = pipeline.inference_ar_and_fm(
            src_wav_path=src_path,
            timbre_ref_wav_path=ref_path,
            style_ref_wav_path=style_path,
            flow_matching_steps=steps,
        )
        return gen_audio.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert_voice(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        reference_audio: np.ndarray,
        reference_sr: int,
        mode: str = "timbre",
        flow_matching_steps: int = 32,
        style_reference_audio: Optional[np.ndarray] = None,
        style_reference_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert *source_audio* using *reference_audio* as the target voice.

        Args:
            source_audio: Source waveform as float32 numpy array (mono, 1-D).
            source_sr: Sample rate of *source_audio*.
            reference_audio: Reference waveform (target voice).
            reference_sr: Sample rate of *reference_audio*.
            mode: ``"timbre"`` for style-preserved VC (FM only) or
                  ``"voice"`` for full VC (AR + FM).
            flow_matching_steps: Number of flow-matching diffusion steps (1-64).
            style_reference_audio: Optional separate style reference for "voice" mode.
                                   If *None* in "voice" mode, *reference_audio* is reused.
            style_reference_sr: Sample rate of *style_reference_audio*.

        Returns:
            Tuple of ``(converted_audio, sample_rate)`` where
            *converted_audio* is a float32 numpy array and *sample_rate* is 24000.
        """
        src_tmp = None
        ref_tmp = None
        style_tmp = None

        try:
            # Write source audio to temporary WAV
            src_tmp = self._write_temp_wav(source_audio, source_sr)
            ref_tmp = self._write_temp_wav(reference_audio, reference_sr)

            if mode == "timbre":
                result = self._convert_timbre(src_tmp, ref_tmp, steps=flow_matching_steps)
            elif mode == "voice":
                # Determine style reference
                if style_reference_audio is not None and style_reference_sr is not None:
                    style_tmp = self._write_temp_wav(style_reference_audio, style_reference_sr)
                else:
                    style_tmp = ref_tmp  # reuse reference as style
                result = self._convert_voice(src_tmp, ref_tmp, style_tmp, steps=flow_matching_steps)
            else:
                raise ValueError(f"[VEVO] Unknown mode '{mode}'. Use 'timbre' or 'voice'.")

            return result, OUTPUT_SAMPLE_RATE

        finally:
            # Clean up temp files
            for path in (src_tmp, ref_tmp, style_tmp):
                if path is not None and os.path.isfile(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _write_temp_wav(audio: np.ndarray, sr: int) -> str:
        """Write numpy audio to a temporary WAV file and return its path."""
        tensor = torch.from_numpy(audio).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # (channels, samples)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        torchaudio.save(path, tensor, sr)
        return path
