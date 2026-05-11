"""
Seed-VC Engine - Zero-shot voice conversion using Seed-VC models.

Supports V1 (offline/realtime) and V2 variants from Plachta/Seed-VC on HuggingFace.
Models are lazily downloaded and loaded on first inference call.
"""

import os
import sys
import tempfile
import numpy as np
import torch
from typing import Tuple, Optional

import comfy.model_management as model_management

# Add project root for imports
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class SeedVCEngine:
    """
    Core Seed-VC voice conversion engine.

    Provides lazy model loading and inference for Seed-VC V1 (offline/realtime)
    and V2 variants.  Models are downloaded from HuggingFace on first use and
    cached under ``folder_paths.models_dir / TTS / Seed-VC /``.
    """

    # HuggingFace repo for model weights
    HF_REPO_ID = "Plachta/Seed-VC"

    # Default output sample rate (both V1 and V2 output 22050 Hz)
    OUTPUT_SR = 22050

    def __init__(
        self,
        variant: str = "v2",
        diffusion_steps: int = 25,
        length_adjust: float = 1.0,
        intelligibility_cfg_rate: float = 0.7,
        similarity_cfg_rate: float = 0.7,
        convert_style: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialise engine parameters.  No models are loaded here.

        Args:
            variant: Model variant - ``v2``, ``v1_offline``, or ``v1_realtime``.
            diffusion_steps: Number of diffusion steps (1-50).
            length_adjust: Length adjustment factor (0.5-2.0).
            intelligibility_cfg_rate: CFG rate for intelligibility (0.0-1.0).
            similarity_cfg_rate: CFG rate for speaker similarity (0.0-1.0).
            convert_style: Whether to transfer speaking style from source.
            device: Torch device string.  ``None`` uses ComfyUI default.
        """
        self.variant = variant
        self.diffusion_steps = diffusion_steps
        self.length_adjust = length_adjust
        self.intelligibility_cfg_rate = intelligibility_cfg_rate
        self.similarity_cfg_rate = similarity_cfg_rate
        self.convert_style = convert_style

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = model_management.get_torch_device()

        # Lazy-loaded model references
        self._model = None
        self._model_loaded_variant: Optional[str] = None
        self._model_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device):
        """Move loaded model components to *device*."""
        self.device = torch.device(device) if isinstance(device, str) else device
        if self._model is not None and hasattr(self._model, "to"):
            self._model.to(self.device)
        return self

    # ------------------------------------------------------------------
    # Model downloading / path resolution
    # ------------------------------------------------------------------

    def _get_model_dir(self) -> str:
        """Return (and create) the local cache directory for Seed-VC weights."""
        if self._model_dir is not None:
            return self._model_dir

        try:
            import folder_paths
            base = os.path.join(folder_paths.models_dir, "TTS", "Seed-VC")
        except ImportError:
            base = os.path.join(tempfile.gettempdir(), "Seed-VC")

        os.makedirs(base, exist_ok=True)
        self._model_dir = base
        return base

    def _ensure_downloaded(self) -> str:
        """
        Download the Seed-VC snapshot from HuggingFace if not already present.

        Returns:
            Path to the local snapshot directory.
        """
        model_dir = self._get_model_dir()

        # Quick check: if key config files already exist, skip download
        v2_config = os.path.join(model_dir, "vc_wrapper.yaml")
        v1_config = os.path.join(model_dir, "config.yml")
        if os.path.isfile(v2_config) or os.path.isfile(v1_config):
            return model_dir

        print(f"Downloading Seed-VC models from {self.HF_REPO_ID} ...")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=self.HF_REPO_ID,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Seed-VC models saved to {model_dir}")
        return model_dir

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the model for the configured variant (lazy, once)."""
        if self._model is not None and self._model_loaded_variant == self.variant:
            return

        model_dir = self._ensure_downloaded()

        print(f"Loading Seed-VC model (variant={self.variant}) on {self.device} ...")

        if self.variant == "v2":
            self._load_v2(model_dir)
        elif self.variant in ("v1_offline", "v1_realtime"):
            self._load_v1(model_dir)
        else:
            raise ValueError(f"Unknown Seed-VC variant: {self.variant}")

        self._model_loaded_variant = self.variant
        print(f"Seed-VC model loaded (variant={self.variant})")

    def _load_v2(self, model_dir: str):
        """Load V2 model via hydra instantiation of ``vc_wrapper.yaml``."""
        config_path = os.path.join(model_dir, "vc_wrapper.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"V2 config not found at {config_path}. "
                "Re-download the model or check your Seed-VC installation."
            )

        from omegaconf import OmegaConf
        from hydra.utils import instantiate

        cfg = OmegaConf.load(config_path)

        # Resolve any relative paths inside the config so they point at model_dir
        OmegaConf.update(cfg, "model_dir", model_dir, force_add=True)

        model = instantiate(cfg)
        if hasattr(model, "to"):
            model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()

        self._model = model

    def _load_v1(self, model_dir: str):
        """Load V1 model using config YAML + checkpoint."""
        config_path = os.path.join(model_dir, "config.yml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"V1 config not found at {config_path}. "
                "Re-download the model or check your Seed-VC installation."
            )

        from omegaconf import OmegaConf
        from hydra.utils import instantiate

        cfg = OmegaConf.load(config_path)

        # Locate the checkpoint
        ckpt_candidates = [
            os.path.join(model_dir, "v1_offline.pth"),
            os.path.join(model_dir, "v1_realtime.pth"),
            os.path.join(model_dir, "seed_vc.pth"),
        ]

        ckpt_path = None
        for candidate in ckpt_candidates:
            if os.path.isfile(candidate):
                ckpt_path = candidate
                break

        if ckpt_path is None:
            # Try to find any .pth file in the directory
            for f in os.listdir(model_dir):
                if f.endswith(".pth"):
                    ckpt_path = os.path.join(model_dir, f)
                    break

        model = instantiate(cfg)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded V1 checkpoint: {os.path.basename(ckpt_path)}")

        if hasattr(model, "to"):
            model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()

        self._model = model

    # ------------------------------------------------------------------
    # Audio I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_wav(path: str, audio: np.ndarray, sr: int):
        """Write a numpy array to a WAV file (float32, mono)."""
        import scipy.io.wavfile as wavfile

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        # Clip to [-1, 1] for WAV
        audio = np.clip(audio, -1.0, 1.0)
        wavfile.write(path, sr, audio)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def convert_voice(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        reference_audio: np.ndarray,
        reference_sr: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert the voice of *source_audio* to match *reference_audio*.

        Both inputs are 1-D float32 numpy arrays.  Temp WAV files are used
        because Seed-VC's inference API expects file paths.

        Returns:
            Tuple of (converted_audio_float32_1d, output_sample_rate).
        """
        self._load_model()

        source_path = None
        reference_path = None
        output_path = None

        try:
            # Write source and reference to temp WAV files
            source_fd, source_path = tempfile.mkstemp(suffix="_src.wav")
            os.close(source_fd)
            self._write_wav(source_path, source_audio, source_sr)

            ref_fd, reference_path = tempfile.mkstemp(suffix="_ref.wav")
            os.close(ref_fd)
            self._write_wav(reference_path, reference_audio, reference_sr)

            # Prepare output path
            out_fd, output_path = tempfile.mkstemp(suffix="_out.wav")
            os.close(out_fd)

            # --- V2 inference ---
            if self.variant == "v2":
                result = self._infer_v2(source_path, reference_path, output_path)
            # --- V1 inference ---
            else:
                result = self._infer_v1(source_path, reference_path, output_path)

            return result

        finally:
            # Clean up temp files
            for p in (source_path, reference_path, output_path):
                if p is not None:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    def _infer_v2(
        self, source_path: str, reference_path: str, output_path: str
    ) -> Tuple[np.ndarray, int]:
        """Run V2 inference using ``convert_voice_with_streaming``."""
        model = self._model

        if hasattr(model, "convert_voice_with_streaming"):
            # V2 streaming API
            chunks = []
            for chunk in model.convert_voice_with_streaming(
                source=source_path,
                target=reference_path,
                diffusion_steps=self.diffusion_steps,
                length_adjust=self.length_adjust,
                intelligibility_cfg_rate=self.intelligibility_cfg_rate,
                similarity_cfg_rate=self.similarity_cfg_rate,
                convert_style=self.convert_style,
            ):
                if isinstance(chunk, torch.Tensor):
                    chunks.append(chunk.cpu().numpy())
                elif isinstance(chunk, np.ndarray):
                    chunks.append(chunk)

            if chunks:
                audio_out = np.concatenate(
                    [c.squeeze() for c in chunks], axis=-1
                )
            else:
                audio_out = np.zeros(self.OUTPUT_SR, dtype=np.float32)
        elif hasattr(model, "convert_voice"):
            # Fallback non-streaming API
            result = model.convert_voice(
                source=source_path,
                target=reference_path,
                diffusion_steps=self.diffusion_steps,
                length_adjust=self.length_adjust,
            )
            if isinstance(result, torch.Tensor):
                audio_out = result.cpu().numpy().squeeze()
            else:
                audio_out = np.asarray(result, dtype=np.float32).squeeze()
        else:
            raise RuntimeError(
                "Loaded V2 model has no convert_voice or "
                "convert_voice_with_streaming method."
            )

        return audio_out.astype(np.float32), self.OUTPUT_SR

    def _infer_v1(
        self, source_path: str, reference_path: str, output_path: str
    ) -> Tuple[np.ndarray, int]:
        """Run V1 inference using the repo's inference function."""
        model = self._model

        if hasattr(model, "inference"):
            result = model.inference(
                source=source_path,
                target=reference_path,
                diffusion_steps=self.diffusion_steps,
                length_adjust=self.length_adjust,
            )
        elif hasattr(model, "convert_voice"):
            result = model.convert_voice(
                source=source_path,
                target=reference_path,
                diffusion_steps=self.diffusion_steps,
                length_adjust=self.length_adjust,
            )
        elif hasattr(model, "__call__"):
            result = model(
                source=source_path,
                target=reference_path,
                diffusion_steps=self.diffusion_steps,
                length_adjust=self.length_adjust,
            )
        else:
            raise RuntimeError(
                "Loaded V1 model has no inference, convert_voice, or __call__ method."
            )

        if isinstance(result, torch.Tensor):
            audio_out = result.cpu().numpy().squeeze()
        elif isinstance(result, tuple):
            # Some versions return (audio, sr)
            audio_out = result[0]
            if isinstance(audio_out, torch.Tensor):
                audio_out = audio_out.cpu().numpy().squeeze()
            else:
                audio_out = np.asarray(audio_out, dtype=np.float32).squeeze()
        else:
            audio_out = np.asarray(result, dtype=np.float32).squeeze()

        return audio_out.astype(np.float32), self.OUTPUT_SR

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Release model memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded_variant = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Seed-VC engine cleanup completed")
