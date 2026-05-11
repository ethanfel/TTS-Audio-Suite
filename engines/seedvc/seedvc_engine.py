"""
Seed-VC Engine - Zero-shot voice conversion using Seed-VC models.

Clones the seed-vc source repo on first use, loads models via Hydra
config + HuggingFace checkpoints.  Supports V2 (recommended).
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import torch
from typing import Tuple, Optional

import comfy.model_management as model_management

try:
    import folder_paths
except ImportError:
    folder_paths = None

SEEDVC_REPO_URL = "https://github.com/Plachtaa/seed-vc.git"


class SeedVCEngine:
    """
    Core Seed-VC voice conversion engine.

    Clones the seed-vc GitHub repo on first use for model definitions and
    config files, then downloads checkpoints from HuggingFace.
    """

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

        self._model = None
        self._model_loaded_variant: Optional[str] = None
        self._repo_dir: Optional[str] = None
        self._base_dir: Optional[str] = None

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
    # Repo / path management
    # ------------------------------------------------------------------

    def _get_base_dir(self) -> str:
        if self._base_dir is not None:
            return self._base_dir

        if folder_paths is not None:
            base = os.path.join(folder_paths.models_dir, "TTS", "Seed-VC")
        else:
            base = os.path.join(tempfile.gettempdir(), "Seed-VC")

        os.makedirs(base, exist_ok=True)
        self._base_dir = base
        return base

    def _ensure_repo(self) -> str:
        """Clone the seed-vc repo if not already present."""
        if self._repo_dir and os.path.isdir(self._repo_dir):
            return self._repo_dir

        base = self._get_base_dir()
        repo_dir = os.path.join(base, "repo")

        if not os.path.isdir(os.path.join(repo_dir, ".git")):
            print(f"[Seed-VC] Cloning repo to {repo_dir} ...")
            subprocess.check_call(
                ["git", "clone", "--depth", "1", SEEDVC_REPO_URL, repo_dir],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            print("[Seed-VC] Repo cloned.")
        else:
            print(f"[Seed-VC] Repo already present at {repo_dir}")

        self._repo_dir = repo_dir
        return repo_dir

    def _inject_repo_path(self):
        """Ensure the seed-vc source is importable."""
        repo_dir = self._ensure_repo()
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the model for the configured variant (lazy, once)."""
        if self._model is not None and self._model_loaded_variant == self.variant:
            return

        print(f"[Seed-VC] Loading model (variant={self.variant}) on {self.device} ...")

        if self.variant == "v2":
            self._load_v2()
        elif self.variant in ("v1_offline", "v1_realtime"):
            raise NotImplementedError(
                "Seed-VC V1 is not yet supported. Use variant='v2'."
            )
        else:
            raise ValueError(f"Unknown Seed-VC variant: {self.variant}")

        self._model_loaded_variant = self.variant
        print(f"[Seed-VC] Model loaded (variant={self.variant})")

    def _load_v2(self):
        """Load V2 model using the repo's Hydra config + HuggingFace checkpoints."""
        self._inject_repo_path()

        import yaml
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        config_path = os.path.join(self._repo_dir, "configs", "v2", "vc_wrapper.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"V2 config not found at {config_path}. "
                "The seed-vc repo may have changed structure."
            )

        print("[Seed-VC] Instantiating V2 model from config...")
        with open(config_path) as f:
            cfg = DictConfig(yaml.safe_load(f))

        # The seed-vc repo has a top-level 'modules' package that collides with
        # RVC's 'modules.py' already cached in sys.modules.  Clear the stale
        # entries so Hydra's import_module finds the seed-vc one.
        for key in list(sys.modules.keys()):
            if key == "modules" or key.startswith("modules."):
                del sys.modules[key]

        model = instantiate(cfg)

        # load_checkpoints() downloads weights from HuggingFace via hf_hub_download.
        # It writes to ./checkpoints/ relative to CWD, so temporarily change CWD.
        print("[Seed-VC] Downloading checkpoints from HuggingFace...")
        base = self._get_base_dir()
        old_cwd = os.getcwd()
        try:
            os.chdir(base)
            model.load_checkpoints()
        finally:
            os.chdir(old_cwd)

        model.to(self.device)
        model.eval()

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model.setup_ar_caches(
            max_batch_size=1, max_seq_len=4096, dtype=dtype, device=self.device
        )

        self._model = model

    # ------------------------------------------------------------------
    # Audio I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_wav(path: str, audio: np.ndarray, sr: int):
        """Write a numpy array to a WAV file (float32, mono)."""
        import soundfile as sf

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        audio = np.clip(audio, -1.0, 1.0)
        sf.write(path, audio, sr)

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

        try:
            source_fd, source_path = tempfile.mkstemp(suffix="_src.wav")
            os.close(source_fd)
            self._write_wav(source_path, source_audio, source_sr)

            ref_fd, reference_path = tempfile.mkstemp(suffix="_ref.wav")
            os.close(ref_fd)
            self._write_wav(reference_path, reference_audio, reference_sr)

            if self.variant == "v2":
                result = self._infer_v2(source_path, reference_path)
            else:
                raise NotImplementedError("Only V2 is currently supported.")

            return result

        finally:
            for p in (source_path, reference_path):
                if p is not None:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    def _infer_v2(
        self, source_path: str, reference_path: str
    ) -> Tuple[np.ndarray, int]:
        """Run V2 inference."""
        model = self._model

        # stream_output=False returns concatenated numpy array directly
        audio_out = model.convert_voice_with_streaming(
            source_audio_path=source_path,
            target_audio_path=reference_path,
            diffusion_steps=self.diffusion_steps,
            length_adjust=self.length_adjust,
            intelligebility_cfg_rate=self.intelligibility_cfg_rate,
            similarity_cfg_rate=self.similarity_cfg_rate,
            convert_style=self.convert_style,
            device=self.device,
            stream_output=False,
        )
        if isinstance(audio_out, torch.Tensor):
            audio_out = audio_out.cpu().numpy().squeeze()
        else:
            audio_out = np.asarray(audio_out, dtype=np.float32).squeeze()

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
        print("[Seed-VC] Engine cleanup completed")
