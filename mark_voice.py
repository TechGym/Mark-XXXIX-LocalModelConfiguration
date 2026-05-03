"""Local speech-to-text for Ollama mode (separate from Ollama chat weights)."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

_whisper_model = None


def _get_whisper():
    """Lazy-load faster-whisper (downloads weights on first use)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    from faster_whisper import WhisperModel

    size = (os.environ.get("MARK_WHISPER_SIZE", "") or "small").strip().lower()
    if size not in ("tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"):
        size = "small"
    device = (os.environ.get("MARK_WHISPER_DEVICE", "") or "cpu").strip().lower()
    if device not in ("cpu", "cuda"):
        device = "cpu"
    compute_type = os.environ.get("MARK_WHISPER_COMPUTE", "").strip() or (
        "int8" if device == "cpu" else "float16"
    )
    _whisper_model = WhisperModel(size, device=device, compute_type=compute_type)
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

    segments, _info = model.transcribe(
        audio,
        language=lang if lang else None,
        vad_filter=True,
    )
    parts = [s.text.strip() for s in segments if s.text.strip()]
    return " ".join(parts).strip()
