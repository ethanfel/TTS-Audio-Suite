# Seed-VC, EZ-VC, and VEVO Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three zero-shot voice conversion engines (Seed-VC, EZ-VC, VEVO) to the TTS Audio Suite's Unified Voice Changer node, following the existing 6-layer integration pattern.

**Architecture:** Each engine gets: engine implementation (`engines/<name>/`), adapter (`engines/adapters/<name>_adapter.py`), engine node (`nodes/engines/<name>_engine_node.py`), Voice Changer routing, and engine registry entry. The adapter returns `(np.ndarray, int)` from `convert_voice()`. The engine node returns `(adapter,)` as `TTS_ENGINE`. Models auto-download from HuggingFace on first use.

**Tech Stack:** PyTorch, torchaudio, HuggingFace Hub, hydra-core (Seed-VC), XEUS/espnet (EZ-VC), Amphion/Vocos (VEVO), ComfyUI node system.

**Design doc:** `docs/plans/2026-05-11-vc-engines-design.md`

---

## Task 1: Engine Registry — Register All Three Engines

**Files:**
- Modify: `utils/models/engine_registry.py:80-109` (add entries before closing brace)

**Step 1: Add registry entries**

In `utils/models/engine_registry.py`, add after the existing `"granite_asr"` entry (line 108) and before the closing `}`:

```python
    "seedvc": EngineCapabilities(
        supports_voice_conversion=True,
        multilingual_model_switching=False,
        can_corrupt_on_reload=False,
        fallback_languages=[],
    ),

    "ezvc": EngineCapabilities(
        supports_voice_conversion=True,
        multilingual_model_switching=False,
        can_corrupt_on_reload=False,
        fallback_languages=[],
    ),

    "vevo": EngineCapabilities(
        supports_voice_conversion=True,
        multilingual_model_switching=False,
        can_corrupt_on_reload=False,
        fallback_languages=[],
    ),
```

**Step 2: Verify**

Run: `python -c "from utils.models.engine_registry import ENGINE_REGISTRY; print([k for k in ENGINE_REGISTRY if k in ('seedvc','ezvc','vevo')])"`
Expected: `['seedvc', 'ezvc', 'vevo']`

**Step 3: Commit**

```bash
git add utils/models/engine_registry.py
git commit -m "feat: register seedvc, ezvc, vevo in engine registry"
```

---

## Task 2: Seed-VC Engine Implementation

**Files:**
- Create: `engines/seedvc/__init__.py`
- Create: `engines/seedvc/seedvc_engine.py`

**Step 1: Create `engines/seedvc/__init__.py`**

```python
"""Seed-VC zero-shot voice conversion engine."""
```

**Step 2: Create `engines/seedvc/seedvc_engine.py`**

This wraps the Seed-VC inference pipeline. Models auto-download from `Plachta/Seed-VC` on HuggingFace.

```python
"""
Seed-VC Engine - Zero-shot any-to-any voice conversion using Diffusion Transformers.
Supports V1 (offline/realtime) and V2 (style/accent transfer).
Models auto-download from HuggingFace: Plachta/Seed-VC
"""

import os
import sys
import torch
import numpy as np
import tempfile
import soundfile as sf
from typing import Tuple, Optional, Dict, Any

import comfy.model_management as model_management


class SeedVCEngine:
    """Seed-VC voice conversion engine wrapper."""

    HF_REPO = "Plachta/Seed-VC"

    # Model variant configs
    VARIANTS = {
        "v1_offline": {
            "config": "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
            "checkpoint": "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            "sample_rate": 22050,
            "description": "Offline VC (98M params, Whisper-small, BigVGAN)",
        },
        "v1_realtime": {
            "config": "config_dit_mel_seed_uvit_xlsr_tiny.yml",
            "checkpoint": "DiT_uvit_tat_xlsr_ema.pth",
            "sample_rate": 22050,
            "description": "Real-time VC (25M params, XLSR, HiFi-GAN)",
        },
        "v2": {
            "sample_rate": 22050,
            "description": "V2 with style/accent transfer (AR + CFM)",
        },
    }

    def __init__(self):
        self.device = model_management.get_torch_device()
        self.models = {}
        self._current_variant = None
        self._model_dir = None

    def to(self, device):
        self.device = device
        return self

    def _ensure_models_downloaded(self) -> str:
        """Download Seed-VC models from HuggingFace if not present. Returns local dir."""
        if self._model_dir and os.path.exists(self._model_dir):
            return self._model_dir

        try:
            import folder_paths
            base_dir = os.path.join(folder_paths.models_dir, "TTS", "Seed-VC")
        except ImportError:
            base_dir = os.path.join(os.path.expanduser("~"), ".cache", "seed-vc")

        os.makedirs(base_dir, exist_ok=True)

        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id=self.HF_REPO,
            local_dir=base_dir,
            local_dir_use_symlinks=False,
        )
        self._model_dir = local_dir
        return local_dir

    def _load_v1_model(self, variant: str):
        """Load Seed-VC V1 model (offline or realtime)."""
        if self._current_variant == variant and variant in self.models:
            return

        model_dir = self._ensure_models_downloaded()
        variant_cfg = self.VARIANTS[variant]

        # Add seed-vc repo to path for imports
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        import yaml
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        config_path = os.path.join(model_dir, variant_cfg["config"])
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        dit_config = DictConfig(config)
        # Load model through seed-vc's own loading mechanism
        from modules.commons import build_model
        model = build_model(dit_config)

        ckpt_path = os.path.join(model_dir, variant_cfg["checkpoint"])
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model = model.to(self.device).eval()
        self.models[variant] = {"model": model, "config": dit_config}
        self._current_variant = variant
        print(f"✅ Seed-VC {variant} model loaded on {self.device}")

    def _load_v2_model(self):
        """Load Seed-VC V2 model (AR + CFM)."""
        if self._current_variant == "v2" and "v2" in self.models:
            return

        model_dir = self._ensure_models_downloaded()

        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        import yaml
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        config_path = os.path.join(model_dir, "configs", "v2", "vc_wrapper.yaml")
        with open(config_path, "r") as f:
            cfg = DictConfig(yaml.safe_load(f))

        vc_wrapper = instantiate(cfg)
        vc_wrapper.load_checkpoints(ar_checkpoint_path=None, cfm_checkpoint_path=None)
        vc_wrapper.to(self.device)
        vc_wrapper.eval()

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=self.device)

        self.models["v2"] = {"wrapper": vc_wrapper, "dtype": dtype}
        self._current_variant = "v2"
        print(f"✅ Seed-VC V2 model loaded on {self.device}")

    def convert_voice(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        reference_audio: np.ndarray,
        reference_sr: int,
        variant: str = "v2",
        diffusion_steps: int = 25,
        length_adjust: float = 1.0,
        intelligibility_cfg_rate: float = 0.7,
        similarity_cfg_rate: float = 0.7,
        convert_style: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert source audio to match reference speaker voice.

        Returns:
            (converted_audio_np, sample_rate)
        """
        # Write audio to temp files (Seed-VC expects file paths)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src_f:
            sf.write(src_f.name, source_audio, source_sr)
            src_path = src_f.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_f:
            sf.write(ref_f.name, reference_audio, reference_sr)
            ref_path = ref_f.name

        try:
            if variant == "v2":
                return self._convert_v2(
                    src_path, ref_path, diffusion_steps, length_adjust,
                    intelligibility_cfg_rate, similarity_cfg_rate, convert_style,
                )
            else:
                return self._convert_v1(
                    src_path, ref_path, variant, diffusion_steps, length_adjust,
                    intelligibility_cfg_rate,
                )
        finally:
            os.unlink(src_path)
            os.unlink(ref_path)

    def _convert_v1(
        self, src_path, ref_path, variant, diffusion_steps, length_adjust, cfg_rate
    ) -> Tuple[np.ndarray, int]:
        self._load_v1_model(variant)
        model_dir = self._model_dir

        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        from inference import inference_single

        output_sr = self.VARIANTS[variant]["sample_rate"]
        result = inference_single(
            source_path=src_path,
            target_path=ref_path,
            model=self.models[variant]["model"],
            config=self.models[variant]["config"],
            device=self.device,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            inference_cfg_rate=cfg_rate,
        )

        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        if result.ndim > 1:
            result = result.squeeze()

        return result.astype(np.float32), output_sr

    def _convert_v2(
        self, src_path, ref_path, diffusion_steps, length_adjust,
        intelligibility_cfg_rate, similarity_cfg_rate, convert_style,
    ) -> Tuple[np.ndarray, int]:
        self._load_v2_model()
        wrapper = self.models["v2"]["wrapper"]
        dtype = self.models["v2"]["dtype"]

        generator = wrapper.convert_voice_with_streaming(
            source_audio_path=src_path,
            target_audio_path=ref_path,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            intelligebility_cfg_rate=intelligibility_cfg_rate,
            similarity_cfg_rate=similarity_cfg_rate,
            convert_style=convert_style,
            device=self.device,
            dtype=dtype,
            stream_output=False,
        )

        sr, audio = None, None
        for output in generator:
            sr, audio = output

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()

        return audio.astype(np.float32), sr or 22050
```

**Step 3: Verify syntax**

Run: `cd /media/p5/TTS-Audio-Suite && python -c "import ast; ast.parse(open('engines/seedvc/seedvc_engine.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add engines/seedvc/
git commit -m "feat: add Seed-VC engine implementation"
```

---

## Task 3: Seed-VC Adapter

**Files:**
- Create: `engines/adapters/seedvc_adapter.py`

**Step 1: Create the adapter**

```python
"""
Seed-VC Engine Adapter - Standardized interface for Seed-VC voice conversion
in the unified voice changer system.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, Union


class SeedVCEngineAdapter:
    """Engine-specific adapter for Seed-VC voice conversion."""

    def __init__(self, node_instance=None):
        self.node = node_instance
        self.engine_type = "seedvc"
        self._engine = None
        self.config = {}

    def _get_engine(self):
        if self._engine is None:
            from engines.seedvc.seedvc_engine import SeedVCEngine
            self._engine = SeedVCEngine()
        return self._engine

    def convert_voice(
        self,
        source_audio: Dict[str, Any],
        reference_audio: Dict[str, Any],
        **kwargs,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Convert source audio to match reference speaker.

        Args:
            source_audio: ComfyUI AUDIO dict {waveform, sample_rate}
            reference_audio: ComfyUI AUDIO dict {waveform, sample_rate}

        Returns:
            (converted_audio_dict, info_string)
        """
        engine = self._get_engine()

        # Resolve device
        device = self.config.get("device", "auto")
        if device == "auto":
            import comfy.model_management as mm
            device = str(mm.get_torch_device())
        engine.to(torch.device(device))

        # Extract numpy arrays from ComfyUI audio dicts
        src_np, src_sr = self._audio_dict_to_numpy(source_audio)
        ref_np, ref_sr = self._audio_dict_to_numpy(reference_audio)

        # Get engine parameters from config
        variant = self.config.get("model_variant", "v2")
        diffusion_steps = self.config.get("diffusion_steps", 25)
        length_adjust = self.config.get("length_adjust", 1.0)
        intelligibility_cfg_rate = self.config.get("intelligibility_cfg_rate", 0.7)
        similarity_cfg_rate = self.config.get("similarity_cfg_rate", 0.7)
        convert_style = self.config.get("convert_style", False)

        result_np, result_sr = engine.convert_voice(
            source_audio=src_np,
            source_sr=src_sr,
            reference_audio=ref_np,
            reference_sr=ref_sr,
            variant=variant,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            intelligibility_cfg_rate=intelligibility_cfg_rate,
            similarity_cfg_rate=similarity_cfg_rate,
            convert_style=convert_style,
        )

        converted_audio = self._numpy_to_audio_dict(result_np, result_sr)

        info = (
            f"Seed-VC ({variant}) | "
            f"Steps: {diffusion_steps} | "
            f"Style: {'on' if convert_style else 'off'} | "
            f"Device: {device}"
        )

        return converted_audio, info

    def _audio_dict_to_numpy(self, audio_dict: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        waveform = audio_dict["waveform"]
        sr = audio_dict.get("sample_rate", 22050)

        if isinstance(waveform, torch.Tensor):
            if waveform.dtype == torch.bfloat16:
                waveform = waveform.to(torch.float32)
            audio_np = waveform.detach().cpu().numpy()
        else:
            audio_np = np.array(waveform)

        # Ensure 1D mono
        if audio_np.ndim > 1:
            if audio_np.shape[0] <= 2:
                audio_np = audio_np[0]
            elif audio_np.shape[1] <= 2:
                audio_np = audio_np[:, 0]
            else:
                audio_np = audio_np.squeeze()
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)

        return audio_np.astype(np.float32), sr

    def _numpy_to_audio_dict(self, audio_np: np.ndarray, sr: int) -> Dict[str, Any]:
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        if np.max(np.abs(audio_np)) > 1.0:
            audio_np = audio_np / np.max(np.abs(audio_np))

        # ComfyUI format: (batch, channels, samples)
        waveform = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sr}
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('engines/adapters/seedvc_adapter.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add engines/adapters/seedvc_adapter.py
git commit -m "feat: add Seed-VC adapter for unified voice changer"
```

---

## Task 4: Seed-VC Engine Node

**Files:**
- Create: `nodes/engines/seedvc_engine_node.py`

**Step 1: Create the node**

```python
"""
Seed-VC Engine Node - Zero-shot voice conversion using Diffusion Transformers.
Supports V1 (offline/realtime) and V2 (style/accent transfer).
"""

import os
import sys
import importlib.util

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

BaseTTSNode = base_module.BaseTTSNode

from engines.adapters.seedvc_adapter import SeedVCEngineAdapter


class SeedVCEngineNode(BaseTTSNode):
    """
    Seed-VC Engine - Zero-shot voice conversion.

    ⚠️ Voice conversion only - does NOT generate speech from text.
    Use with the Voice Changer node to convert audio.
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
                    "tooltip": "Model variant:\n• v2: Best quality with style/accent transfer (recommended)\n• v1_offline: Standard quality, Whisper encoder\n• v1_realtime: Fast, lightweight (25M params)"
                }),
                "diffusion_steps": ("INT", {
                    "default": 25, "min": 1, "max": 50, "step": 1,
                    "display": "slider",
                    "tooltip": "Number of diffusion steps. Higher = better quality but slower. 10-15 for fast, 25-30 for quality."
                }),
                "length_adjust": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "display": "slider",
                    "tooltip": "Speed adjustment. <1.0 = faster speech, >1.0 = slower speech."
                }),
            },
            "optional": {
                "intelligibility_cfg_rate": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "slider",
                    "tooltip": "(V2 only) Controls output speech clarity. Higher = clearer but less similar to reference."
                }),
                "similarity_cfg_rate": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "slider",
                    "tooltip": "(V2 only) Controls voice similarity to reference. Higher = more similar to reference speaker."
                }),
                "convert_style": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "(V2 only) Enable accent/emotion/style transfer from reference audio."
                }),
                "device": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto",
                    "tooltip": "Processing device. Auto selects best available."
                }),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    CATEGORY = "TTS Audio Suite/⚙️ Engines"
    FUNCTION = "create_engine"

    DESCRIPTION = """
    Seed-VC Engine - Zero-shot Any-to-Any Voice Conversion

    ⚠️ Voice conversion only - does NOT generate speech from text.
    Use with Voice Changer node only.

    Converts any voice to sound like a reference speaker with no training required.
    V2 adds accent, emotion, and style transfer capabilities.
    """

    def create_engine(
        self,
        model_variant="v2",
        diffusion_steps=25,
        length_adjust=1.0,
        intelligibility_cfg_rate=0.7,
        similarity_cfg_rate=0.7,
        convert_style=False,
        device="auto",
    ):
        try:
            adapter = SeedVCEngineAdapter()

            if device == "auto":
                import comfy.model_management as model_management
                device = str(model_management.get_torch_device())

            adapter.config = {
                "type": "seedvc_engine",
                "engine_type": "seedvc",
                "model_variant": model_variant,
                "diffusion_steps": diffusion_steps,
                "length_adjust": length_adjust,
                "intelligibility_cfg_rate": intelligibility_cfg_rate,
                "similarity_cfg_rate": similarity_cfg_rate,
                "convert_style": convert_style,
                "device": device,
            }

            style_info = " (style transfer ON)" if convert_style else ""
            print(f"⚙️ Seed-VC Engine created - Variant: {model_variant}, Steps: {diffusion_steps}{style_info}, Device: {device}")
            return (adapter,)

        except Exception as e:
            print(f"❌ Seed-VC Engine creation failed: {e}")
            return (None,)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('nodes/engines/seedvc_engine_node.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add nodes/engines/seedvc_engine_node.py
git commit -m "feat: add Seed-VC engine node for ComfyUI"
```

---

## Task 5: EZ-VC Engine Implementation

**Files:**
- Create: `engines/ezvc/__init__.py`
- Create: `engines/ezvc/ezvc_engine.py`

**Step 1: Create `engines/ezvc/__init__.py`**

```python
"""EZ-VC zero-shot voice conversion engine."""
```

**Step 2: Create `engines/ezvc/ezvc_engine.py`**

```python
"""
EZ-VC Engine - Zero-shot any-to-any voice conversion using XEUS encoder + F5-TTS decoder.
Single encoder architecture, works across languages.
Models auto-download from HuggingFace: SPRINGLab/EZ-VC
"""

import os
import sys
import re
import torch
import numpy as np
import tempfile
import soundfile as sf
from typing import Tuple, Optional, Dict, Any

import comfy.model_management as model_management


class EZVCEngine:
    """EZ-VC voice conversion engine wrapper."""

    HF_REPO = "SPRINGLab/EZ-VC"
    GITHUB_REPO = "https://github.com/EZ-VC/EZ-VC"

    def __init__(self):
        self.device = model_management.get_torch_device()
        self._model = None
        self._vocoder = None
        self._xeus_model = None
        self._apply_kmeans = None
        self._repo_dir = None
        self._model_dir = None

    def to(self, device):
        self.device = device
        if self._xeus_model is not None:
            self._xeus_model = self._xeus_model.to(device)
        return self

    def _ensure_repo_cloned(self) -> str:
        """Clone EZ-VC repo if not present (needed for inference code). Returns repo dir."""
        if self._repo_dir and os.path.exists(self._repo_dir):
            return self._repo_dir

        try:
            import folder_paths
            base_dir = os.path.join(folder_paths.models_dir, "TTS", "EZ-VC")
        except ImportError:
            base_dir = os.path.join(os.path.expanduser("~"), ".cache", "ez-vc")

        repo_dir = os.path.join(base_dir, "repo")
        if not os.path.exists(os.path.join(repo_dir, "src")):
            import subprocess
            os.makedirs(base_dir, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.GITHUB_REPO, repo_dir],
                check=True,
            )
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=repo_dir,
                check=True,
            )

        self._repo_dir = repo_dir
        return repo_dir

    def _ensure_models_downloaded(self) -> str:
        """Download model weights from HuggingFace. Returns checkpoint path."""
        from cached_path import cached_path
        ckpt_path = str(cached_path(f"hf://{self.HF_REPO}/model_2700000.safetensors"))
        vocab_path = str(cached_path(f"hf://{self.HF_REPO}/vocab.txt"))
        self._model_dir = os.path.dirname(ckpt_path)
        return ckpt_path, vocab_path

    def _load_models(self):
        """Load all EZ-VC models on first use."""
        if self._model is not None:
            return

        repo_dir = self._ensure_repo_cloned()
        src_dir = os.path.join(repo_dir, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.infer.utils_xeus import load_xeus_model, ApplyKmeans
        from hydra.utils import get_class
        from omegaconf import OmegaConf

        # Load XEUS encoder and K-means quantizer
        self._xeus_model = load_xeus_model(self.device).eval()
        self._apply_kmeans = ApplyKmeans(self.device)

        # Load vocoder
        self._vocoder = load_vocoder(vocoder_name="bigvgan", device=self.device)

        # Load F5-TTS decoder
        ckpt_path, vocab_path = self._ensure_models_downloaded()
        config_path = os.path.join(repo_dir, "src", "f5_tts", "configs", "F5TTS_Base_EZ-VC.yaml")

        model_cfg = OmegaConf.load(config_path)
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        self._model = load_model(
            model_cls, model_arc, ckpt_path, mel_spec_type="bigvgan",
            vocab_file=vocab_path, device=self.device,
        )

        print(f"✅ EZ-VC models loaded on {self.device}")

    def convert_voice(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        reference_audio: np.ndarray,
        reference_sr: int,
        nfe_steps: int = 12,
        speed: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert source audio to match reference speaker.

        Returns:
            (converted_audio_np, sample_rate)
        """
        self._load_models()

        repo_dir = self._repo_dir
        src_dir = os.path.join(repo_dir, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from f5_tts.infer.utils_infer import (
            infer_process, cfg_strength, cross_fade_duration,
            fix_duration, sway_sampling_coef, target_rms,
        )
        from f5_tts.infer.utils_xeus import extract_units

        # Write to temp files for XEUS processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src_f:
            sf.write(src_f.name, source_audio, source_sr)
            src_path = src_f.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_f:
            sf.write(ref_f.name, reference_audio, reference_sr)
            ref_path = ref_f.name

        try:
            # Extract speech units
            ref_text = extract_units(ref_path, self._xeus_model, self._apply_kmeans, self.device)
            src_text = extract_units(src_path, self._xeus_model, self._apply_kmeans, self.device)

            # Run inference
            generated_segments = []
            reg1 = r"(?=\[\w+\])"
            chunks = re.split(reg1, src_text)
            reg2 = r"\[(\w+)\]"

            for text in chunks:
                text = re.sub(reg2, "", text).strip()
                if not text:
                    continue
                audio_segment, final_sample_rate, _ = infer_process(
                    ref_path, ref_text, text, self._model, self._vocoder,
                    mel_spec_type="bigvgan", target_rms=target_rms,
                    cross_fade_duration=cross_fade_duration, nfe_step=nfe_steps,
                    cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef,
                    speed=speed, fix_duration=fix_duration, device=self.device,
                )
                generated_segments.append(audio_segment)

            if generated_segments:
                result = np.concatenate(generated_segments)
            else:
                result = source_audio
                final_sample_rate = source_sr

            return result.astype(np.float32), final_sample_rate or 16000

        finally:
            os.unlink(src_path)
            os.unlink(ref_path)
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('engines/ezvc/ezvc_engine.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add engines/ezvc/
git commit -m "feat: add EZ-VC engine implementation"
```

---

## Task 6: EZ-VC Adapter

**Files:**
- Create: `engines/adapters/ezvc_adapter.py`

**Step 1: Create the adapter**

```python
"""
EZ-VC Engine Adapter - Standardized interface for EZ-VC voice conversion
in the unified voice changer system.
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple


class EZVCEngineAdapter:
    """Engine-specific adapter for EZ-VC voice conversion."""

    def __init__(self, node_instance=None):
        self.node = node_instance
        self.engine_type = "ezvc"
        self._engine = None
        self.config = {}

    def _get_engine(self):
        if self._engine is None:
            from engines.ezvc.ezvc_engine import EZVCEngine
            self._engine = EZVCEngine()
        return self._engine

    def convert_voice(
        self,
        source_audio: Dict[str, Any],
        reference_audio: Dict[str, Any],
        **kwargs,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Convert source audio to match reference speaker.

        Args:
            source_audio: ComfyUI AUDIO dict {waveform, sample_rate}
            reference_audio: ComfyUI AUDIO dict {waveform, sample_rate}

        Returns:
            (converted_audio_dict, info_string)
        """
        engine = self._get_engine()

        device = self.config.get("device", "auto")
        if device == "auto":
            import comfy.model_management as mm
            device = str(mm.get_torch_device())
        engine.to(torch.device(device))

        src_np, src_sr = self._audio_dict_to_numpy(source_audio)
        ref_np, ref_sr = self._audio_dict_to_numpy(reference_audio)

        nfe_steps = self.config.get("nfe_steps", 12)
        speed = self.config.get("speed", 1.0)

        result_np, result_sr = engine.convert_voice(
            source_audio=src_np,
            source_sr=src_sr,
            reference_audio=ref_np,
            reference_sr=ref_sr,
            nfe_steps=nfe_steps,
            speed=speed,
        )

        converted_audio = self._numpy_to_audio_dict(result_np, result_sr)

        info = (
            f"EZ-VC | "
            f"NFE Steps: {nfe_steps} | "
            f"Speed: {speed}x | "
            f"Device: {device}"
        )

        return converted_audio, info

    def _audio_dict_to_numpy(self, audio_dict: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        waveform = audio_dict["waveform"]
        sr = audio_dict.get("sample_rate", 16000)

        if isinstance(waveform, torch.Tensor):
            if waveform.dtype == torch.bfloat16:
                waveform = waveform.to(torch.float32)
            audio_np = waveform.detach().cpu().numpy()
        else:
            audio_np = np.array(waveform)

        if audio_np.ndim > 1:
            if audio_np.shape[0] <= 2:
                audio_np = audio_np[0]
            elif audio_np.shape[1] <= 2:
                audio_np = audio_np[:, 0]
            else:
                audio_np = audio_np.squeeze()
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)

        return audio_np.astype(np.float32), sr

    def _numpy_to_audio_dict(self, audio_np: np.ndarray, sr: int) -> Dict[str, Any]:
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        if np.max(np.abs(audio_np)) > 1.0:
            audio_np = audio_np / np.max(np.abs(audio_np))

        waveform = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sr}
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('engines/adapters/ezvc_adapter.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add engines/adapters/ezvc_adapter.py
git commit -m "feat: add EZ-VC adapter for unified voice changer"
```

---

## Task 7: EZ-VC Engine Node

**Files:**
- Create: `nodes/engines/ezvc_engine_node.py`

**Step 1: Create the node**

```python
"""
EZ-VC Engine Node - Zero-shot voice conversion using XEUS encoder + F5-TTS decoder.
Single encoder, cross-lingual, textless voice conversion.
"""

import os
import sys
import importlib.util

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

BaseTTSNode = base_module.BaseTTSNode

from engines.adapters.ezvc_adapter import EZVCEngineAdapter


class EZVCEngineNode(BaseTTSNode):
    """
    EZ-VC Engine - Zero-shot voice conversion.

    ⚠️ Voice conversion only - does NOT generate speech from text.
    Use with the Voice Changer node to convert audio.
    """

    @classmethod
    def NAME(cls):
        return "⚙️ EZ-VC Engine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nfe_steps": ("INT", {
                    "default": 12, "min": 1, "max": 32, "step": 1,
                    "display": "slider",
                    "tooltip": "Number of flow-matching steps. Higher = better quality but slower. 8-12 for fast, 16-24 for quality."
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "display": "slider",
                    "tooltip": "Speech speed. <1.0 = faster, >1.0 = slower."
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto",
                    "tooltip": "Processing device. Auto selects best available."
                }),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    CATEGORY = "TTS Audio Suite/⚙️ Engines"
    FUNCTION = "create_engine"

    DESCRIPTION = """
    EZ-VC Engine - Easy Zero-shot Voice Conversion

    ⚠️ Voice conversion only - does NOT generate speech from text.
    Use with Voice Changer node only.

    Uses a single XEUS encoder with F5-TTS decoder for cross-lingual
    zero-shot voice conversion. Works across languages including
    ones never seen during training.
    """

    def create_engine(self, nfe_steps=12, speed=1.0, device="auto"):
        try:
            adapter = EZVCEngineAdapter()

            if device == "auto":
                import comfy.model_management as model_management
                device = str(model_management.get_torch_device())

            adapter.config = {
                "type": "ezvc_engine",
                "engine_type": "ezvc",
                "nfe_steps": nfe_steps,
                "speed": speed,
                "device": device,
            }

            print(f"⚙️ EZ-VC Engine created - NFE Steps: {nfe_steps}, Speed: {speed}x, Device: {device}")
            return (adapter,)

        except Exception as e:
            print(f"❌ EZ-VC Engine creation failed: {e}")
            return (None,)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('nodes/engines/ezvc_engine_node.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add nodes/engines/ezvc_engine_node.py
git commit -m "feat: add EZ-VC engine node for ComfyUI"
```

---

## Task 8: VEVO Engine Implementation

**Files:**
- Create: `engines/vevo/__init__.py`
- Create: `engines/vevo/vevo_engine.py`

**Step 1: Create `engines/vevo/__init__.py`**

```python
"""VEVO zero-shot voice imitation engine (from Amphion)."""
```

**Step 2: Create `engines/vevo/vevo_engine.py`**

```python
"""
VEVO Engine - Zero-shot voice imitation with self-supervised disentanglement.
Supports Timbre-only VC and full Voice conversion (timbre + style).
Models auto-download from HuggingFace: amphion/Vevo
From the Amphion toolkit (open-mmlab/Amphion).
"""

import os
import sys
import torch
import numpy as np
import tempfile
import soundfile as sf
from typing import Tuple, Optional, Dict, Any

import comfy.model_management as model_management


class VevoEngine:
    """VEVO voice conversion engine wrapper."""

    HF_REPO = "amphion/Vevo"
    AMPHION_REPO = "https://github.com/open-mmlab/Amphion"

    def __init__(self):
        self.device = model_management.get_torch_device()
        self._pipeline_timbre = None
        self._pipeline_voice = None
        self._amphion_dir = None

    def to(self, device):
        self.device = device
        return self

    def _ensure_amphion_cloned(self) -> str:
        """Clone Amphion repo for VEVO inference code. Returns Amphion dir."""
        if self._amphion_dir and os.path.exists(self._amphion_dir):
            return self._amphion_dir

        try:
            import folder_paths
            base_dir = os.path.join(folder_paths.models_dir, "TTS", "VEVO")
        except ImportError:
            base_dir = os.path.join(os.path.expanduser("~"), ".cache", "vevo")

        amphion_dir = os.path.join(base_dir, "Amphion")
        if not os.path.exists(os.path.join(amphion_dir, "models", "vc", "vevo")):
            import subprocess
            os.makedirs(base_dir, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.AMPHION_REPO, amphion_dir],
                check=True,
            )

        self._amphion_dir = amphion_dir
        return amphion_dir

    def _download_model_component(self, pattern: str) -> str:
        """Download a specific VEVO model component from HuggingFace."""
        from huggingface_hub import snapshot_download

        try:
            import folder_paths
            cache_dir = os.path.join(folder_paths.models_dir, "TTS", "VEVO", "ckpts")
        except ImportError:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vevo", "ckpts")

        local_dir = snapshot_download(
            repo_id=self.HF_REPO,
            repo_type="model",
            cache_dir=cache_dir,
            allow_patterns=[f"{pattern}/*"],
        )
        return os.path.join(local_dir, pattern)

    def _load_timbre_pipeline(self):
        """Load VEVO Timbre pipeline (tokenizer + flow-matching + vocoder)."""
        if self._pipeline_timbre is not None:
            return

        amphion_dir = self._ensure_amphion_cloned()
        if amphion_dir not in sys.path:
            sys.path.insert(0, amphion_dir)

        from models.vc.vevo.vevo_utils import VevoInferencePipeline

        tokenizer_path = self._download_model_component("tokenizer/vq8192")
        fmt_ckpt_path = self._download_model_component("acoustic_modeling/Vq8192ToMels")
        vocoder_ckpt_path = self._download_model_component("acoustic_modeling/Vocoder")

        config_dir = os.path.join(amphion_dir, "models", "vc", "vevo", "config")

        self._pipeline_timbre = VevoInferencePipeline(
            content_style_tokenizer_ckpt_path=tokenizer_path,
            fmt_cfg_path=os.path.join(config_dir, "Vq8192ToMels.json"),
            fmt_ckpt_path=fmt_ckpt_path,
            vocoder_cfg_path=os.path.join(config_dir, "Vocoder.json"),
            vocoder_ckpt_path=vocoder_ckpt_path,
            device=self.device,
        )
        print(f"✅ VEVO Timbre pipeline loaded on {self.device}")

    def _load_voice_pipeline(self):
        """Load VEVO Voice pipeline (content tokenizer + AR + flow-matching + vocoder)."""
        if self._pipeline_voice is not None:
            return

        amphion_dir = self._ensure_amphion_cloned()
        if amphion_dir not in sys.path:
            sys.path.insert(0, amphion_dir)

        from models.vc.vevo.vevo_utils import VevoInferencePipeline

        content_tokenizer_path = self._download_model_component("tokenizer/vq32")
        tokenizer_path = self._download_model_component("tokenizer/vq8192")
        ar_ckpt_path = self._download_model_component("contentstyle_modeling/Vq32ToVq8192")
        fmt_ckpt_path = self._download_model_component("acoustic_modeling/Vq8192ToMels")
        vocoder_ckpt_path = self._download_model_component("acoustic_modeling/Vocoder")

        config_dir = os.path.join(amphion_dir, "models", "vc", "vevo", "config")

        self._pipeline_voice = VevoInferencePipeline(
            content_tokenizer_ckpt_path=content_tokenizer_path,
            content_style_tokenizer_ckpt_path=tokenizer_path,
            ar_cfg_path=os.path.join(config_dir, "Vq32ToVq8192.json"),
            ar_ckpt_path=ar_ckpt_path,
            fmt_cfg_path=os.path.join(config_dir, "Vq8192ToMels.json"),
            fmt_ckpt_path=fmt_ckpt_path,
            vocoder_cfg_path=os.path.join(config_dir, "Vocoder.json"),
            vocoder_ckpt_path=vocoder_ckpt_path,
            device=self.device,
        )
        print(f"✅ VEVO Voice pipeline loaded on {self.device}")

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
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert source audio to match reference speaker.

        Args:
            mode: "timbre" (style-preserved) or "voice" (full conversion)
            style_reference_audio: Optional separate style reference for "voice" mode

        Returns:
            (converted_audio_np, sample_rate)
        """
        # Write audio to temp files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src_f:
            sf.write(src_f.name, source_audio, source_sr)
            src_path = src_f.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_f:
            sf.write(ref_f.name, reference_audio, reference_sr)
            ref_path = ref_f.name

        style_path = None
        if style_reference_audio is not None:
            style_f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(style_f.name, style_reference_audio, style_reference_sr or reference_sr)
            style_path = style_f.name
            style_f.close()

        try:
            if mode == "timbre":
                return self._convert_timbre(src_path, ref_path, flow_matching_steps)
            else:
                effective_style_path = style_path or ref_path
                return self._convert_voice(src_path, ref_path, effective_style_path, flow_matching_steps)
        finally:
            os.unlink(src_path)
            os.unlink(ref_path)
            if style_path:
                os.unlink(style_path)

    def _convert_timbre(self, src_path, ref_path, steps) -> Tuple[np.ndarray, int]:
        self._load_timbre_pipeline()

        amphion_dir = self._amphion_dir
        if amphion_dir not in sys.path:
            sys.path.insert(0, amphion_dir)

        from models.vc.vevo.vevo_utils import save_audio

        gen_audio = self._pipeline_timbre.inference_fm(
            src_wav_path=src_path,
            timbre_ref_wav_path=ref_path,
            flow_matching_steps=steps,
        )

        if isinstance(gen_audio, torch.Tensor):
            gen_audio = gen_audio.cpu().numpy()
        if gen_audio.ndim > 1:
            gen_audio = gen_audio.squeeze()

        return gen_audio.astype(np.float32), 24000

    def _convert_voice(self, src_path, ref_path, style_path, steps) -> Tuple[np.ndarray, int]:
        self._load_voice_pipeline()

        amphion_dir = self._amphion_dir
        if amphion_dir not in sys.path:
            sys.path.insert(0, amphion_dir)

        gen_audio = self._pipeline_voice.inference_ar_and_fm(
            src_wav_path=src_path,
            src_text=None,
            style_ref_wav_path=style_path,
            timbre_ref_wav_path=ref_path,
            flow_matching_steps=steps,
        )

        if isinstance(gen_audio, torch.Tensor):
            gen_audio = gen_audio.cpu().numpy()
        if gen_audio.ndim > 1:
            gen_audio = gen_audio.squeeze()

        return gen_audio.astype(np.float32), 24000
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('engines/vevo/vevo_engine.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add engines/vevo/
git commit -m "feat: add VEVO engine implementation"
```

---

## Task 9: VEVO Adapter

**Files:**
- Create: `engines/adapters/vevo_adapter.py`

**Step 1: Create the adapter**

```python
"""
VEVO Engine Adapter - Standardized interface for VEVO voice conversion
in the unified voice changer system.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple


class VevoEngineAdapter:
    """Engine-specific adapter for VEVO voice conversion."""

    def __init__(self, node_instance=None):
        self.node = node_instance
        self.engine_type = "vevo"
        self._engine = None
        self.config = {}

    def _get_engine(self):
        if self._engine is None:
            from engines.vevo.vevo_engine import VevoEngine
            self._engine = VevoEngine()
        return self._engine

    def convert_voice(
        self,
        source_audio: Dict[str, Any],
        reference_audio: Dict[str, Any],
        style_reference_audio: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Convert source audio to match reference speaker.

        Args:
            source_audio: ComfyUI AUDIO dict {waveform, sample_rate}
            reference_audio: ComfyUI AUDIO dict {waveform, sample_rate}
            style_reference_audio: Optional separate style reference for "voice" mode

        Returns:
            (converted_audio_dict, info_string)
        """
        engine = self._get_engine()

        device = self.config.get("device", "auto")
        if device == "auto":
            import comfy.model_management as mm
            device = str(mm.get_torch_device())
        engine.to(torch.device(device))

        src_np, src_sr = self._audio_dict_to_numpy(source_audio)
        ref_np, ref_sr = self._audio_dict_to_numpy(reference_audio)

        style_np, style_sr = None, None
        if style_reference_audio is not None:
            style_np, style_sr = self._audio_dict_to_numpy(style_reference_audio)

        mode = self.config.get("mode", "timbre")
        flow_matching_steps = self.config.get("flow_matching_steps", 32)

        result_np, result_sr = engine.convert_voice(
            source_audio=src_np,
            source_sr=src_sr,
            reference_audio=ref_np,
            reference_sr=ref_sr,
            mode=mode,
            flow_matching_steps=flow_matching_steps,
            style_reference_audio=style_np,
            style_reference_sr=style_sr,
        )

        converted_audio = self._numpy_to_audio_dict(result_np, result_sr)

        info = (
            f"VEVO ({mode}) | "
            f"Steps: {flow_matching_steps} | "
            f"{'Separate style ref' if style_np is not None else 'Same ref for timbre+style'} | "
            f"Device: {device}"
        )

        return converted_audio, info

    def _audio_dict_to_numpy(self, audio_dict: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        waveform = audio_dict["waveform"]
        sr = audio_dict.get("sample_rate", 24000)

        if isinstance(waveform, torch.Tensor):
            if waveform.dtype == torch.bfloat16:
                waveform = waveform.to(torch.float32)
            audio_np = waveform.detach().cpu().numpy()
        else:
            audio_np = np.array(waveform)

        if audio_np.ndim > 1:
            if audio_np.shape[0] <= 2:
                audio_np = audio_np[0]
            elif audio_np.shape[1] <= 2:
                audio_np = audio_np[:, 0]
            else:
                audio_np = audio_np.squeeze()
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)

        return audio_np.astype(np.float32), sr

    def _numpy_to_audio_dict(self, audio_np: np.ndarray, sr: int) -> Dict[str, Any]:
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        if np.max(np.abs(audio_np)) > 1.0:
            audio_np = audio_np / np.max(np.abs(audio_np))

        waveform = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sr}
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('engines/adapters/vevo_adapter.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add engines/adapters/vevo_adapter.py
git commit -m "feat: add VEVO adapter for unified voice changer"
```

---

## Task 10: VEVO Engine Node

**Files:**
- Create: `nodes/engines/vevo_engine_node.py`

**Step 1: Create the node**

```python
"""
VEVO Engine Node - Zero-shot voice imitation with timbre/style disentanglement.
Supports timbre-only VC and full voice conversion (timbre + style).
"""

import os
import sys
import importlib.util

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

BaseTTSNode = base_module.BaseTTSNode

from engines.adapters.vevo_adapter import VevoEngineAdapter


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


class VevoEngineNode(BaseTTSNode):
    """
    VEVO Engine - Zero-shot voice imitation.

    ⚠️ Voice conversion only - does NOT generate speech from text.
    Use with the Voice Changer node to convert audio.
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
                    "tooltip": "Conversion mode:\n• timbre: Change speaker identity only (preserves original speaking style, accent, emotion)\n• voice: Full conversion - change both timbre AND style to match reference"
                }),
                "flow_matching_steps": ("INT", {
                    "default": 32, "min": 1, "max": 64, "step": 1,
                    "display": "slider",
                    "tooltip": "Number of flow-matching steps. Higher = better quality but slower. 16-24 for fast, 32 for quality."
                }),
            },
            "optional": {
                "style_reference": (any_typ, {
                    "tooltip": "(Voice mode only) Separate style reference audio. If not connected, uses the same reference as narrator_target in Voice Changer."
                }),
                "device": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto",
                    "tooltip": "Processing device. Auto selects best available."
                }),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    CATEGORY = "TTS Audio Suite/⚙️ Engines"
    FUNCTION = "create_engine"

    DESCRIPTION = """
    VEVO Engine - Zero-shot Voice Imitation (ICLR 2025)

    ⚠️ Voice conversion only - does NOT generate speech from text.
    Use with Voice Changer node only.

    Disentangles timbre, style, and content using self-supervised VQ-VAE bottlenecks.
    Timbre mode preserves original speaking style. Voice mode transfers both identity and style.
    Optional separate style reference allows mixing timbre from one speaker with style from another.
    """

    def create_engine(self, mode="timbre", flow_matching_steps=32, style_reference=None, device="auto"):
        try:
            adapter = VevoEngineAdapter()

            if device == "auto":
                import comfy.model_management as model_management
                device = str(model_management.get_torch_device())

            adapter.config = {
                "type": "vevo_engine",
                "engine_type": "vevo",
                "mode": mode,
                "flow_matching_steps": flow_matching_steps,
                "device": device,
            }

            # Store style reference in adapter if provided
            if style_reference is not None:
                adapter._style_reference = style_reference

            print(f"⚙️ VEVO Engine created - Mode: {mode}, Steps: {flow_matching_steps}, Device: {device}")
            return (adapter,)

        except Exception as e:
            print(f"❌ VEVO Engine creation failed: {e}")
            return (None,)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('nodes/engines/vevo_engine_node.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add nodes/engines/vevo_engine_node.py
git commit -m "feat: add VEVO engine node for ComfyUI"
```

---

## Task 11: Register Nodes in nodes.py

**Files:**
- Modify: `nodes.py`

**Step 1: Add node loading**

After the RVC engine loading block (around line 332), add:

```python
# Load Seed-VC nodes
try:
    seedvc_engine_module = load_node_module("seedvc_engine_node", "engines/seedvc_engine_node.py")
    SeedVCEngineNode = seedvc_engine_module.SeedVCEngineNode
    SEEDVC_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"❌ Seed-VC Engine failed: {e}")
    SEEDVC_ENGINE_AVAILABLE = False

# Load EZ-VC nodes
try:
    ezvc_engine_module = load_node_module("ezvc_engine_node", "engines/ezvc_engine_node.py")
    EZVCEngineNode = ezvc_engine_module.EZVCEngineNode
    EZVC_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"❌ EZ-VC Engine failed: {e}")
    EZVC_ENGINE_AVAILABLE = False

# Load VEVO nodes
try:
    vevo_engine_module = load_node_module("vevo_engine_node", "engines/vevo_engine_node.py")
    VevoEngineNode = vevo_engine_module.VevoEngineNode
    VEVO_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"❌ VEVO Engine failed: {e}")
    VEVO_ENGINE_AVAILABLE = False
```

**Step 2: Add node registration**

After the RVC engine registration block (around line 649), add:

```python
if SEEDVC_ENGINE_AVAILABLE:
    NODE_CLASS_MAPPINGS["SeedVCEngineNode"] = SeedVCEngineNode
    NODE_DISPLAY_NAME_MAPPINGS["SeedVCEngineNode"] = "⚙️ Seed-VC Engine"

if EZVC_ENGINE_AVAILABLE:
    NODE_CLASS_MAPPINGS["EZVCEngineNode"] = EZVCEngineNode
    NODE_DISPLAY_NAME_MAPPINGS["EZVCEngineNode"] = "⚙️ EZ-VC Engine"

if VEVO_ENGINE_AVAILABLE:
    NODE_CLASS_MAPPINGS["VevoEngineNode"] = VevoEngineNode
    NODE_DISPLAY_NAME_MAPPINGS["VevoEngineNode"] = "⚙️ VEVO Engine"
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('nodes.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add nodes.py
git commit -m "feat: register Seed-VC, EZ-VC, VEVO nodes in node discovery"
```

---

## Task 12: Voice Changer — Add Engine Routing

**Files:**
- Modify: `nodes/unified/voice_changer_node.py`

This is the most critical task. Three changes needed:

**Step 1: Extend the supported engines list**

In `convert_voice()` (around line 888), change the engine validation:

```python
# OLD:
if engine_type not in ["chatterbox", "chatterbox_official_23lang", "rvc", "cosyvoice"]:
    raise ValueError(f"Engine '{engine_type}' does not support voice conversion. Currently supported engines: ChatterBox, ChatterBox Official 23-Lang, RVC, CosyVoice")

# NEW:
if engine_type not in ["chatterbox", "chatterbox_official_23lang", "rvc", "cosyvoice", "seedvc", "ezvc", "vevo"]:
    raise ValueError(f"Engine '{engine_type}' does not support voice conversion. Currently supported engines: ChatterBox, ChatterBox Official 23-Lang, RVC, CosyVoice, Seed-VC, EZ-VC, VEVO")
```

**Step 2: Add adapter routing in `convert_voice()`**

At the top of `convert_voice()`, right after the RVC adapter check (line 865-868), add checks for the new adapters:

```python
            # Check for Seed-VC adapter
            if hasattr(TTS_engine, 'engine_type') and TTS_engine.engine_type == "seedvc":
                return self._handle_seedvc_conversion(
                    TTS_engine, source_audio, narrator_target, refinement_passes,
                    max_chunk_duration, chunk_method)

            # Check for EZ-VC adapter
            if hasattr(TTS_engine, 'engine_type') and TTS_engine.engine_type == "ezvc":
                return self._handle_ezvc_conversion(
                    TTS_engine, source_audio, narrator_target, refinement_passes,
                    max_chunk_duration, chunk_method)

            # Check for VEVO adapter
            if hasattr(TTS_engine, 'engine_type') and TTS_engine.engine_type == "vevo":
                return self._handle_vevo_conversion(
                    TTS_engine, source_audio, narrator_target, refinement_passes,
                    max_chunk_duration, chunk_method)
```

**Step 3: Add the handler methods**

Add the following methods to `UnifiedVoiceChangerNode` class (before `_generate_rvc_cache_key`):

```python
    def _handle_seedvc_conversion(self, adapter, source_audio, narrator_target,
                                   refinement_passes, max_chunk_duration, chunk_method):
        """Handle Seed-VC voice conversion with chunking support."""
        from utils.audio.chunk_combiner import ChunkCombiner

        processed_source = self._extract_audio_from_input(source_audio, "source_audio")
        processed_target = self._extract_audio_from_input(narrator_target, "narrator_target")

        source_sr = processed_source.get("sample_rate", 22050)
        source_waveform = processed_source.get("waveform")
        if source_waveform is None:
            raise ValueError("Source audio missing waveform data")

        source_chunks = self._split_audio_into_chunks(
            source_waveform, source_sr, max_chunk_duration, chunk_method)

        config = getattr(adapter, 'config', {})
        total_chunks = len(source_chunks)
        print(f"🔄 Voice Changer: Seed-VC conversion ({total_chunks} chunk(s), {refinement_passes} pass(es))")

        converted_chunks = []
        for i, chunk in enumerate(source_chunks, 1):
            chunk_audio = {"waveform": chunk, "sample_rate": source_sr}
            current = chunk_audio

            for p in range(refinement_passes):
                if refinement_passes > 1:
                    print(f"  🔄 Chunk {i}/{total_chunks}, pass {p+1}/{refinement_passes}...")
                result, info = adapter.convert_voice(
                    source_audio=current, reference_audio=processed_target)
                current = result

            converted_chunks.append(current["waveform"])

        output_sr = current.get("sample_rate", 22050)
        if total_chunks > 1:
            combined = ChunkCombiner.combine_chunks(
                converted_chunks, method="crossfade",
                crossfade_duration=0.05, sample_rate=output_sr)
        else:
            combined = converted_chunks[0]

        converted_audio = {"waveform": combined, "sample_rate": output_sr}

        variant = config.get("model_variant", "v2")
        steps = config.get("diffusion_steps", 25)
        chunking_info = f"Chunking: {total_chunks} chunks ({chunk_method} mode, {max_chunk_duration}s max) | " if total_chunks > 1 else ""
        conversion_info = (
            f"🔄 Voice Changer (Unified) - SEED-VC Engine:\n"
            f"Variant: {variant} | Steps: {steps} | "
            f"{chunking_info}"
            f"Refinement passes: {refinement_passes} | "
            f"Device: {config.get('device', 'auto')}"
        )

        print(f"✅ Seed-VC conversion completed")
        return (converted_audio, conversion_info)

    def _handle_ezvc_conversion(self, adapter, source_audio, narrator_target,
                                 refinement_passes, max_chunk_duration, chunk_method):
        """Handle EZ-VC voice conversion with chunking support."""
        from utils.audio.chunk_combiner import ChunkCombiner

        processed_source = self._extract_audio_from_input(source_audio, "source_audio")
        processed_target = self._extract_audio_from_input(narrator_target, "narrator_target")

        source_sr = processed_source.get("sample_rate", 16000)
        source_waveform = processed_source.get("waveform")
        if source_waveform is None:
            raise ValueError("Source audio missing waveform data")

        source_chunks = self._split_audio_into_chunks(
            source_waveform, source_sr, max_chunk_duration, chunk_method)

        config = getattr(adapter, 'config', {})
        total_chunks = len(source_chunks)
        print(f"🔄 Voice Changer: EZ-VC conversion ({total_chunks} chunk(s), {refinement_passes} pass(es))")

        converted_chunks = []
        for i, chunk in enumerate(source_chunks, 1):
            chunk_audio = {"waveform": chunk, "sample_rate": source_sr}
            current = chunk_audio

            for p in range(refinement_passes):
                if refinement_passes > 1:
                    print(f"  🔄 Chunk {i}/{total_chunks}, pass {p+1}/{refinement_passes}...")
                result, info = adapter.convert_voice(
                    source_audio=current, reference_audio=processed_target)
                current = result

            converted_chunks.append(current["waveform"])

        output_sr = current.get("sample_rate", 16000)
        if total_chunks > 1:
            combined = ChunkCombiner.combine_chunks(
                converted_chunks, method="crossfade",
                crossfade_duration=0.05, sample_rate=output_sr)
        else:
            combined = converted_chunks[0]

        converted_audio = {"waveform": combined, "sample_rate": output_sr}

        nfe = config.get("nfe_steps", 12)
        speed = config.get("speed", 1.0)
        chunking_info = f"Chunking: {total_chunks} chunks ({chunk_method} mode, {max_chunk_duration}s max) | " if total_chunks > 1 else ""
        conversion_info = (
            f"🔄 Voice Changer (Unified) - EZ-VC Engine:\n"
            f"NFE Steps: {nfe} | Speed: {speed}x | "
            f"{chunking_info}"
            f"Refinement passes: {refinement_passes} | "
            f"Device: {config.get('device', 'auto')}"
        )

        print(f"✅ EZ-VC conversion completed")
        return (converted_audio, conversion_info)

    def _handle_vevo_conversion(self, adapter, source_audio, narrator_target,
                                 refinement_passes, max_chunk_duration, chunk_method):
        """Handle VEVO voice conversion with chunking support."""
        from utils.audio.chunk_combiner import ChunkCombiner

        processed_source = self._extract_audio_from_input(source_audio, "source_audio")
        processed_target = self._extract_audio_from_input(narrator_target, "narrator_target")

        # Check for separate style reference stored on adapter
        style_ref = None
        if hasattr(adapter, '_style_reference') and adapter._style_reference is not None:
            try:
                style_ref = self._extract_audio_from_input(adapter._style_reference, "style_reference")
            except Exception:
                style_ref = None

        source_sr = processed_source.get("sample_rate", 24000)
        source_waveform = processed_source.get("waveform")
        if source_waveform is None:
            raise ValueError("Source audio missing waveform data")

        source_chunks = self._split_audio_into_chunks(
            source_waveform, source_sr, max_chunk_duration, chunk_method)

        config = getattr(adapter, 'config', {})
        total_chunks = len(source_chunks)
        mode = config.get("mode", "timbre")
        print(f"🔄 Voice Changer: VEVO {mode} conversion ({total_chunks} chunk(s), {refinement_passes} pass(es))")

        converted_chunks = []
        for i, chunk in enumerate(source_chunks, 1):
            chunk_audio = {"waveform": chunk, "sample_rate": source_sr}
            current = chunk_audio

            for p in range(refinement_passes):
                if refinement_passes > 1:
                    print(f"  🔄 Chunk {i}/{total_chunks}, pass {p+1}/{refinement_passes}...")
                result, info = adapter.convert_voice(
                    source_audio=current, reference_audio=processed_target,
                    style_reference_audio=style_ref)
                current = result

            converted_chunks.append(current["waveform"])

        output_sr = current.get("sample_rate", 24000)
        if total_chunks > 1:
            combined = ChunkCombiner.combine_chunks(
                converted_chunks, method="crossfade",
                crossfade_duration=0.05, sample_rate=output_sr)
        else:
            combined = converted_chunks[0]

        converted_audio = {"waveform": combined, "sample_rate": output_sr}

        steps = config.get("flow_matching_steps", 32)
        chunking_info = f"Chunking: {total_chunks} chunks ({chunk_method} mode, {max_chunk_duration}s max) | " if total_chunks > 1 else ""
        style_info = "Separate style ref" if style_ref else "Same ref for timbre+style"
        conversion_info = (
            f"🔄 Voice Changer (Unified) - VEVO Engine:\n"
            f"Mode: {mode} | Steps: {steps} | {style_info} | "
            f"{chunking_info}"
            f"Refinement passes: {refinement_passes} | "
            f"Device: {config.get('device', 'auto')}"
        )

        print(f"✅ VEVO conversion completed")
        return (converted_audio, conversion_info)
```

**Step 4: Update Voice Changer node tooltip**

In `INPUT_TYPES()` (around line 66-67), update the TTS_engine tooltip to mention the new engines:

```python
# OLD:
"tooltip": "TTS/VC engine configuration. Supports ChatterBox TTS Engine, CosyVoice Engine, and RVC Engine for voice conversion."

# NEW:
"tooltip": "TTS/VC engine configuration. Supports ChatterBox, CosyVoice, RVC, Seed-VC, EZ-VC, and VEVO engines for voice conversion."
```

**Step 5: Verify syntax**

Run: `python -c "import ast; ast.parse(open('nodes/unified/voice_changer_node.py').read()); print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add nodes/unified/voice_changer_node.py
git commit -m "feat: add Seed-VC, EZ-VC, VEVO routing in unified voice changer"
```

---

## Task 13: Update Engine Documentation YAML

**Files:**
- Modify: `docs/Dev reports/tts_audio_suite_engines.yaml` (if exists)

**Step 1: Check if YAML exists**

Run: `ls "docs/Dev reports/tts_audio_suite_engines.yaml" 2>/dev/null && echo "exists" || echo "skip"`

If it exists, add entries for the three new engines following the existing format. If not, skip this task.

**Step 2: Commit if changes made**

```bash
git add "docs/Dev reports/"
git commit -m "docs: add Seed-VC, EZ-VC, VEVO to engine documentation"
```

---

## Task 14: Integration Test — Verify Node Loading

**Step 1: Test that all new nodes load without errors**

Run from the TTS Audio Suite root:

```bash
python -c "
import sys, os
sys.path.insert(0, '.')

# Test engine implementations parse
import ast
for f in [
    'engines/seedvc/seedvc_engine.py',
    'engines/ezvc/ezvc_engine.py',
    'engines/vevo/vevo_engine.py',
    'engines/adapters/seedvc_adapter.py',
    'engines/adapters/ezvc_adapter.py',
    'engines/adapters/vevo_adapter.py',
    'nodes/engines/seedvc_engine_node.py',
    'nodes/engines/ezvc_engine_node.py',
    'nodes/engines/vevo_engine_node.py',
]:
    ast.parse(open(f).read())
    print(f'✅ {f} syntax OK')

# Test registry
from utils.models.engine_registry import ENGINE_REGISTRY
for name in ['seedvc', 'ezvc', 'vevo']:
    assert name in ENGINE_REGISTRY, f'{name} missing from registry'
    assert ENGINE_REGISTRY[name].supports_voice_conversion, f'{name} should support VC'
    print(f'✅ {name} registered with VC support')

print('\\n✅ All integration checks passed')
"
```

Expected: All 12 checks pass.

**Step 2: Final commit**

```bash
git add -A
git commit -m "feat: complete Seed-VC, EZ-VC, VEVO integration into unified voice changer"
```

---

## Summary of All Files

| Action | File | Purpose |
|--------|------|---------|
| Create | `engines/seedvc/__init__.py` | Package init |
| Create | `engines/seedvc/seedvc_engine.py` | Seed-VC model loading + inference |
| Create | `engines/ezvc/__init__.py` | Package init |
| Create | `engines/ezvc/ezvc_engine.py` | EZ-VC model loading + inference |
| Create | `engines/vevo/__init__.py` | Package init |
| Create | `engines/vevo/vevo_engine.py` | VEVO model loading + inference |
| Create | `engines/adapters/seedvc_adapter.py` | Standardized VC adapter |
| Create | `engines/adapters/ezvc_adapter.py` | Standardized VC adapter |
| Create | `engines/adapters/vevo_adapter.py` | Standardized VC adapter |
| Create | `nodes/engines/seedvc_engine_node.py` | ComfyUI engine node |
| Create | `nodes/engines/ezvc_engine_node.py` | ComfyUI engine node |
| Create | `nodes/engines/vevo_engine_node.py` | ComfyUI engine node |
| Modify | `utils/models/engine_registry.py` | Register 3 engines |
| Modify | `nodes.py` | Load + register 3 engine nodes |
| Modify | `nodes/unified/voice_changer_node.py` | Route to 3 new engines |
