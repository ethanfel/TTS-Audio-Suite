"""
EZ-VC Engine - Zero-shot voice conversion using XEUS encoder and F5-TTS decoder.
Downloads and manages the EZ-VC repo and model weights, runs inference.
"""

import os
import sys
import tempfile
import subprocess
import warnings
from typing import Tuple, Optional

import numpy as np
import torch

import comfy.model_management as model_management

try:
    import folder_paths
except ImportError:
    folder_paths = None


class EZVCEngine:
    """
    EZ-VC voice conversion engine.

    Clones the EZ-VC inference repo on first use, downloads model weights via
    ``cached_path``, and runs voice conversion through the XEUS + F5-TTS pipeline.

    Models are loaded lazily on the first call to :meth:`convert_voice`.
    """

    # HuggingFace paths for model artefacts
    HF_WEIGHTS = "hf://SPRINGLab/EZ-VC/model_2700000.safetensors"
    HF_VOCAB = "hf://SPRINGLab/EZ-VC/Emilia_ZH_EN_pinyin/vocab.txt"
    REPO_URL = "https://github.com/EZ-VC/EZ-VC.git"

    # Output sample rate produced by the EZ-VC pipeline
    OUTPUT_SR = 16000

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = model_management.get_torch_device()
        else:
            self.device = torch.device(device)

        self._models_loaded = False

        # Model handles (populated by _load_models)
        self._xeus_model = None
        self._kmeans = None
        self._vocoder = None
        self._f5tts_model = None
        self._vocab_path = None

        # Lazy-import references (populated after repo is available)
        self._load_model = None
        self._load_vocoder = None
        self._infer_process = None
        self._load_xeus_model = None
        self._ApplyKmeans = None
        self._extract_units = None

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device) -> "EZVCEngine":
        """Move the engine to *device*."""
        self.device = torch.device(device) if isinstance(device, str) else device
        if self._models_loaded:
            if self._xeus_model is not None:
                self._xeus_model = self._xeus_model.to(self.device)
            if self._f5tts_model is not None:
                try:
                    self._f5tts_model = self._f5tts_model.to(self.device)
                except Exception:
                    pass  # some wrappers don't support .to()
        return self

    # ------------------------------------------------------------------
    # Repo / dependency bootstrap
    # ------------------------------------------------------------------

    @staticmethod
    def _repo_root() -> str:
        """Return the path where the EZ-VC repo should live."""
        if folder_paths is not None:
            base = folder_paths.models_dir
        else:
            base = os.path.join(os.path.expanduser("~"), "models")
        return os.path.join(base, "TTS", "EZ-VC", "repo")

    def _ensure_repo(self) -> str:
        """Clone the EZ-VC repo if it is not already present and return its path."""
        repo_dir = self._repo_root()
        if os.path.isdir(os.path.join(repo_dir, "src")):
            return repo_dir
        os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
        print(f"[EZ-VC] Cloning repo to {repo_dir} ...")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", self.REPO_URL, repo_dir],
        )
        print("[EZ-VC] Repo cloned successfully.")
        return repo_dir

    def _ensure_repo_on_path(self) -> str:
        """Make sure the repo's ``src`` directory is importable."""
        repo_dir = self._ensure_repo()
        src_dir = os.path.join(repo_dir, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        return repo_dir

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_models(self):
        """Download weights and load all sub-models."""
        if self._models_loaded:
            return

        self._ensure_repo_on_path()

        # --- imports from the cloned repo --------------------------------
        from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process
        from f5_tts.infer.utils_xeus import load_xeus_model, ApplyKmeans, extract_units

        self._load_model = load_model
        self._load_vocoder = load_vocoder
        self._infer_process = infer_process
        self._load_xeus_model = load_xeus_model
        self._ApplyKmeans = ApplyKmeans
        self._extract_units = extract_units

        # --- download weights via cached_path ----------------------------
        from cached_path import cached_path

        print("[EZ-VC] Downloading / caching model weights ...")
        weights_path = str(cached_path(self.HF_WEIGHTS))
        self._vocab_path = str(cached_path(self.HF_VOCAB))

        # --- XEUS encoder ------------------------------------------------
        print("[EZ-VC] Loading XEUS model ...")
        self._xeus_model = load_xeus_model(device=self.device)

        # --- K-means quantiser -------------------------------------------
        print("[EZ-VC] Loading K-means quantiser ...")
        self._kmeans = ApplyKmeans(device=self.device)

        # --- BigVGAN vocoder ---------------------------------------------
        print("[EZ-VC] Loading BigVGAN vocoder ...")
        self._vocoder = load_vocoder(vocoder_name="bigvgan", device=self.device)

        # --- F5-TTS decoder ----------------------------------------------
        print("[EZ-VC] Loading F5-TTS decoder ...")
        self._f5tts_model = load_model(
            model_cls="DiT",
            model_cfg=dict(
                dim=1024,
                depth=22,
                heads=16,
                ff_mult=2,
                text_dim=512,
                conv_layers=4,
            ),
            ckpt_path=weights_path,
            vocab_file=self._vocab_path,
            device=self.device,
        )

        self._models_loaded = True
        print("[EZ-VC] All models loaded.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def convert_voice(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        reference_audio: np.ndarray,
        reference_sr: int,
        nfe_steps: int = 12,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert the voice in *source_audio* to match *reference_audio*.

        Parameters
        ----------
        source_audio : np.ndarray
            1-D float32 source waveform.
        source_sr : int
            Sample rate of *source_audio*.
        reference_audio : np.ndarray
            1-D float32 reference / target-speaker waveform.
        reference_sr : int
            Sample rate of *reference_audio*.
        nfe_steps : int
            Number of flow-matching steps (higher = better quality, slower).
        speed : float
            Playback speed multiplier.

        Returns
        -------
        tuple[np.ndarray, int]
            ``(converted_audio_1d_float32, sample_rate)``
        """
        self._load_models()

        import soundfile as sf

        src_tmp = None
        ref_tmp = None
        try:
            # --- write temp WAVs for the EZ-VC pipeline -------------------
            src_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(src_tmp.name, source_audio, source_sr)
            src_tmp.close()

            ref_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(ref_tmp.name, reference_audio, reference_sr)
            ref_tmp.close()

            # --- extract speech units from source via XEUS ----------------
            source_units = self._extract_units(
                self._xeus_model,
                self._kmeans,
                src_tmp.name,
                device=self.device,
            )

            # --- run F5-TTS infer_process ---------------------------------
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                segments = self._infer_process(
                    ref_audio=ref_tmp.name,
                    ref_text="",
                    gen_text=source_units,
                    model_obj=self._f5tts_model,
                    vocoder=self._vocoder,
                    device=self.device,
                    nfe_step=nfe_steps,
                    speed=speed,
                )

            # --- concatenate returned segments ----------------------------
            audio_chunks = []
            for seg in segments:
                # infer_process yields (sr, audio_np) tuples or similar
                if isinstance(seg, tuple):
                    _sr, chunk = seg
                else:
                    chunk = seg
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.cpu().numpy()
                chunk = np.asarray(chunk, dtype=np.float32).ravel()
                audio_chunks.append(chunk)

            if not audio_chunks:
                raise RuntimeError("EZ-VC infer_process returned no audio segments.")

            converted = np.concatenate(audio_chunks)
            return converted, self.OUTPUT_SR

        finally:
            # --- clean up temp files --------------------------------------
            for tmp in (src_tmp, ref_tmp):
                if tmp is not None:
                    try:
                        os.unlink(tmp.name)
                    except OSError:
                        pass
