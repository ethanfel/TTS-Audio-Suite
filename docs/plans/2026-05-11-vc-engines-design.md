# Design: Adding Seed-VC, EZ-VC, and VEVO Voice Conversion Engines

**Date**: 2026-05-11
**Status**: Approved

## Summary

Add three zero-shot voice conversion engines (Seed-VC, EZ-VC, VEVO) to the Unified Voice Changer using the established 6-layer integration pattern (engine implementation → adapter → engine node → Voice Changer routing → engine registry).

## Source Repositories

| Engine | Repo | HuggingFace | License |
|--------|------|-------------|---------|
| Seed-VC | Plachtaa/seed-vc | Plachta/Seed-VC | GPL-3.0 |
| EZ-VC | EZ-VC/EZ-VC | SPRINGLab/EZ-VC | MIT code / CC-BY-NC-4.0 weights |
| VEVO | open-mmlab/Amphion | amphion/Vevo | MIT code / CC-BY-NC-4.0 weights |

## Architecture

Each engine follows the same pattern as RVC/ChatterBox:

```
⚙️ [Engine] Engine Node  →  TTS_ENGINE dict (engine_type + config)
    ↓
🔄 Voice Changer (unified)  →  routes to engine handler
    ↓
Engine Handler (chunking, refinement)  →  calls adapter
    ↓
[Engine] Adapter  →  standardized convert_voice() interface
    ↓
[Engine] Implementation  →  actual model inference
```

## File Layout

```
engines/
├── seedvc/
│   ├── __init__.py
│   └── seedvc_engine.py          # Model loading, inference wrapper
├── ezvc/
│   ├── __init__.py
│   └── ezvc_engine.py            # XEUS + F5-TTS inference wrapper
├── vevo/
│   ├── __init__.py
│   └── vevo_engine.py            # Amphion pipeline wrapper
└── adapters/
    ├── seedvc_adapter.py         # Standardized VC interface
    ├── ezvc_adapter.py
    └── vevo_adapter.py

nodes/engines/
├── seedvc_engine_node.py         # ComfyUI node with engine-specific params
├── ezvc_engine_node.py
└── vevo_engine_node.py
```

## Engine Node Parameters

### Seed-VC Engine Node (`⚙️ Seed-VC Engine`)

- `model_variant`: ["v1_offline", "v1_realtime", "v2"] (default: "v2")
- `diffusion_steps`: INT 1-50 (default: 25)
- `intelligibility_cfg_rate`: FLOAT 0.0-1.0 (default: 0.7) — v2 only
- `similarity_cfg_rate`: FLOAT 0.0-1.0 (default: 0.7) — v2 only
- `convert_style`: BOOLEAN (default: False) — v2 accent/emotion transfer
- `length_adjust`: FLOAT 0.5-2.0 (default: 1.0)
- `device`: ["auto", "cuda", "cpu"]

### EZ-VC Engine Node (`⚙️ EZ-VC Engine`)

- `nfe_steps`: INT 1-32 (default: 12) — flow-matching steps
- `speed`: FLOAT 0.5-2.0 (default: 1.0)
- `device`: ["auto", "cuda", "cpu"]

### VEVO Engine Node (`⚙️ VEVO Engine`)

- `mode`: ["timbre", "voice"] (default: "timbre")
  - "timbre" = style-preserved VC (only changes speaker identity)
  - "voice" = full VC (changes both timbre and style)
- `flow_matching_steps`: INT 1-64 (default: 32)
- `style_reference`: optional AUDIO input (for "voice" mode; defaults to narrator_target)
- `device`: ["auto", "cuda", "cpu"]

## Model Storage

```
ComfyUI/models/TTS/
├── Seed-VC/           # Auto-downloaded from HF: Plachta/Seed-VC
├── EZ-VC/             # Auto-downloaded from HF: SPRINGLab/EZ-VC
└── VEVO/              # Auto-downloaded from HF: amphion/Vevo
```

## Engine Registry

```python
"seedvc": EngineCapabilities(
    supports_voice_conversion=True,
    multilingual_model_switching=False,
),
"ezvc": EngineCapabilities(
    supports_voice_conversion=True,
    multilingual_model_switching=False,
),
"vevo": EngineCapabilities(
    supports_voice_conversion=True,
    multilingual_model_switching=False,
),
```

## Voice Changer Integration

- Extend `convert_voice()` in `voice_changer_node.py` to support `"seedvc"`, `"ezvc"`, `"vevo"` engine types
- Each engine gets a `_handle_[engine]_conversion()` method
- Chunking and refinement passes handled by Voice Changer, not the engine
- `_create_proper_engine_node_instance()` updated with engine-specific adapter creation

## Inference Pipelines

**Seed-VC**: Source WAV + Reference WAV → Whisper/HuBERT encoding → DiT diffusion → BigVGAN vocoder → output (22/44.1kHz)

**EZ-VC**: Source WAV + Reference WAV → XEUS → K-means quantization → F5-TTS flow matching → BigVGAN → output (16kHz)

**VEVO**: Source WAV + Reference WAV → VQ tokenization → (optional AR for style) → Flow-matching → Vocos vocoder → output (24kHz)

All outputs resampled to match source audio sample rate for consistency.

## Dependencies

Lazy imports per engine with graceful fallback. Engine nodes only appear in ComfyUI when dependencies are available.

### Seed-VC
- hydra-core, descript-audio-codec, modelscope, resemblyzer, munch, einops

### EZ-VC
- espnet (specific fork: wanchichen/espnet@ssl), cached_path, x_transformers, vocos, torchdiffeq, ema_pytorch

### VEVO
- encodec, phonemizer (optional for TTS mode), vocos

Common (already in project): torch, torchaudio, transformers, librosa, soundfile, numpy, huggingface_hub
