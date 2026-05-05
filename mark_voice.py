"""Local speech-to-text for Ollama mode (separate from Ollama chat weights).

Env tuning (PTT latency):

- ``MARK_WHISPER_DEVICE`` — ``auto`` (default): CUDA if a GPU is visible to ctranslate2,
  else CPU. Set ``cpu`` or ``cuda`` to force.
- ``MARK_WHISPER_SIZE`` — ``tiny`` / ``base`` / ``small`` (default) / … Smaller = faster STT.
- ``MARK_WHISPER_BEAM_SIZE`` — default ``1`` (faster than multi-beam).
- ``MARK_WHISPER_CPU_THREADS`` — optional integer for CPU decode threads.
"""

from __future__ import annotations

import os

# Before NumPy / faster-whisper: avoid OMP duplicate-runtime abort on Windows (see main.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Optional

import numpy as np

_whisper_model = None


def _cuda_device_count() -> int:
    try:
        import ctranslate2 as ct2

        fn = getattr(ct2, "get_cuda_device_count", None)
        if callable(fn):
            return int(fn())
    except Exception:
        pass
    return 0


def _resolve_whisper_device(raw: str) -> str:
    """
    ``MARK_WHISPER_DEVICE``: ``cpu`` | ``cuda`` | ``auto`` (default).
    ``auto`` uses CUDA when ctranslate2 sees a GPU, else CPU.
    """
    d = (raw or "").strip().lower()
    if d in ("cpu", "cuda"):
        return d
    return "cuda" if _cuda_device_count() > 0 else "cpu"


def _get_whisper():
    """Lazy-load faster-whisper (downloads weights on first use)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    from faster_whisper import WhisperModel

    size = (os.environ.get("MARK_WHISPER_SIZE", "") or "small").strip().lower()
    if size not in ("tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"):
        size = "small"
    device = _resolve_whisper_device(
        (os.environ.get("MARK_WHISPER_DEVICE", "") or "auto").strip()
    )
    compute_type = os.environ.get("MARK_WHISPER_COMPUTE", "").strip() or (
        "int8" if device == "cpu" else "float16"
    )
    ctor_kw: dict = {}
    ct = (os.environ.get("MARK_WHISPER_CPU_THREADS", "") or "").strip()
    if ct.isdigit() and int(ct) > 0:
        ctor_kw["cpu_threads"] = int(ct)
    _whisper_model = WhisperModel(
        size,
        device=device,
        compute_type=compute_type,
        **ctor_kw,
    )
    print(f"[Voice] 🎙 faster-whisper model={size} device={device} compute={compute_type}")
    return _whisper_model


def transcribe_pcm_int16(pcm: bytes, sample_rate: int, *, language: Optional[str] = None) -> str:
    """
    Transcribe raw PCM int16 mono bytes (e.g. from sounddevice).

    ``language`` defaults to env ``MARK_WHISPER_LANGUAGE`` or ``en``.
    """
    if not pcm or len(pcm) < 1000:
        return ""
    model = _get_whisper()
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    lang = (
        (language or "").strip()
        or os.environ.get("MARK_WHISPER_LANGUAGE", "").strip()
        or "en"
    )
    if sample_rate != 16000:
        # Whisper expects 16 kHz; linear resample if the capture rate differs.
        new_len = max(1, int(len(audio) * 16000 / sample_rate))
        xi = np.linspace(0.0, float(len(audio) - 1), new_len, dtype=np.float64)
        x = np.arange(len(audio), dtype=np.float64)
        audio = np.interp(xi, x, audio.astype(np.float64)).astype(np.float32)

    beam_raw = (os.environ.get("MARK_WHISPER_BEAM_SIZE", "") or "1").strip()
    beam_size = int(beam_raw) if beam_raw.isdigit() else 1
    beam_size = max(1, min(beam_size, 5))

    segments, _info = model.transcribe(
        audio,
        language=lang if lang else None,
        vad_filter=True,
        beam_size=beam_size,
    )
    parts = [s.text.strip() for s in segments if s.text.strip()]
    return " ".join(parts).strip()
