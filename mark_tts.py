"""
Text-to-speech for **local Ollama** reply paths.

1. **Gemini TTS** (optional): set ``tts_backend`` to ``gemini`` in ``api_keys.json`` and
   add ``gemini_api_key``. Uses ``google.genai`` ``generate_content`` with a TTS
   model; audio plays via **sounddevice** (same PortAudio routing as ``main.py``).
2. **pyttsx3 / SAPI** (default): Windows installed voices; SAPI uses the **Windows
   default playback device**.
"""

from __future__ import annotations

import base64
import io
import os
import platform
import sys
import threading
import wave

_tls = threading.local()
# pyttsx3 / SAPI are not safe across thread-pool threads; serialize all TTS.
_tts_lock = threading.Lock()
_tts_backend_hint_printed = False


def _ensure_win32_com_apartment() -> None:
    """SAPI/pyttsx3 often runs from a thread pool; COM must be initialized per thread."""
    if sys.platform != "win32":
        return
    if getattr(_tls, "mark_pyttsx3_com", False):
        return
    try:
        import pythoncom  # type: ignore[import-untyped]
    except ImportError:
        _tls.mark_pyttsx3_com = True
        return
    try:
        pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
    except Exception:
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
    _tls.mark_pyttsx3_com = True


def create_pyttsx3_engine():
    import pyttsx3

    if platform.system() == "Windows":
        for driver in ("sapi5", None):
            try:
                return pyttsx3.init(driver) if driver else pyttsx3.init()
            except Exception:
                continue
    return pyttsx3.init()


def _apply_pyttsx3_voice_substring(engine, substring: str | None) -> None:
    if not substring or not substring.strip():
        return
    needle = substring.strip().lower()
    try:
        voices = engine.getProperty("voices")
    except Exception:
        voices = None
    if not voices:
        return
    for v in voices:
        name = getattr(v, "name", "") or ""
        vid = getattr(v, "id", None)
        if not vid:
            continue
        if needle in name.lower():
            try:
                engine.setProperty("voice", vid)
                print(f"[TTS] Voice: {name!r}")
            except Exception as ex:
                print(f"[TTS] Could not set voice {name!r}: {ex}")
            return
    print(
        f"[TTS] No SAPI voice contains {substring!r}. "
        "List names: python -c \"import pyttsx3; e=pyttsx3.init(); "
        "print([getattr(x,'name',x) for x in (e.getProperty('voices') or [])])\""
    )


def configure_pyttsx3(engine) -> None:
    try:
        engine.setProperty("volume", 1.0)
    except Exception:
        pass
    try:
        rate = engine.getProperty("rate")
        if rate is not None and rate > 260:
            engine.setProperty("rate", 220)
    except Exception:
        pass
    try:
        from mark_llm_settings import get_local_tts_voice_substring

        _apply_pyttsx3_voice_substring(engine, get_local_tts_voice_substring())
    except Exception as ex:
        print(f"[TTS] Voice config skipped: {ex}")


def _speak_pyttsx3_no_lock(text: str) -> None:
    """``pyttsx3`` path; caller must hold ``_tts_lock``."""
    utter = (text or "").strip()
    if not utter:
        return
    _ensure_win32_com_apartment()
    engine = create_pyttsx3_engine()
    configure_pyttsx3(engine)
    engine.say(utter)
    engine.runAndWait()


def _play_pcm_int16_mono(pcm: bytes, sample_rate: int) -> None:
    import numpy as np
    import sounddevice as sd

    audio = np.frombuffer(pcm, dtype=np.int16)
    sd.play(audio, sample_rate, blocking=True)


def _play_wav_bytes_riff(wav_bytes: bytes) -> None:
    """Play RIFF WAV via PortAudio."""
    import numpy as np
    import sounddevice as sd

    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        audio = (audio.astype(np.int32) * 256).astype(np.int16)
    elif sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16)
    elif sw == 4:
        audio = (np.frombuffer(raw, dtype=np.int32) / 65536.0).astype(np.int16)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sw}")
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1).astype(np.int16)
    sd.play(audio, sr, blocking=True)


def _normalize_audio_blob_data(raw) -> bytes | None:
    """API may return ``bytes``, ``memoryview``, or **base64 ``str``** (per REST docs)."""
    if raw is None:
        return None
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    if isinstance(raw, bytes):
        return raw if raw else None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            dec = base64.b64decode(s, validate=False)
        except Exception:
            return None
        return dec if dec else None
    return None


def _gemini_extract_audio(response) -> tuple[bytes, str] | None:
    """Return ``(pcm_or_wav_bytes, mime_type_lower)`` from a ``generate_content`` response."""
    cands = getattr(response, "candidates", None) or []
    audio_parts: list[tuple[bytes, str]] = []
    for cand in cands:
        content = getattr(cand, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        for p in parts:
            inl = getattr(p, "inline_data", None)
            if not inl:
                continue
            mime = (getattr(inl, "mime_type", None) or "").lower()
            blob = _normalize_audio_blob_data(getattr(inl, "data", None))
            if not blob:
                continue
            if "audio" in mime or mime.endswith("/wav") or mime.endswith("/x-wav"):
                audio_parts.append((blob, mime))
            elif not mime and blob[:4] == b"RIFF":
                audio_parts.append((blob, "audio/wav"))
            elif not mime:
                audio_parts.append((blob, "audio/pcm"))
    if audio_parts:
        for blob, mime in audio_parts:
            if "wav" in mime or blob[:4] == b"RIFF":
                return blob, mime
        return audio_parts[0][0], audio_parts[0][1]
    # Fallback: first inline blob (some responses omit audio/* in mime_type)
    for cand in cands:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for p in getattr(content, "parts", None) or []:
            inl = getattr(p, "inline_data", None)
            if not inl:
                continue
            blob = _normalize_audio_blob_data(getattr(inl, "data", None))
            if blob:
                mime = (getattr(inl, "mime_type", None) or "").lower()
                return blob, mime or "audio/pcm"
    return None


_ALT_TTS_MODEL = "gemini-3.1-flash-tts-preview"


def _is_gemini_tts_quota_error(exc: BaseException) -> bool:
    s = str(exc)
    return (
        "429" in s
        or "RESOURCE_EXHAUSTED" in s
        or "Quota exceeded" in s
        or "quota" in s.lower()
    )


def _try_gemini_tts(text: str) -> bool:
    """Synthesize with Gemini TTS API and play. Returns False to fall back to pyttsx3."""
    from mark_llm_settings import (
        get_gemini_api_key,
        get_gemini_live_voice_name,
        get_gemini_tts_model,
        get_local_tts_backend,
    )

    if get_local_tts_backend() != "gemini":
        ev = os.environ.get("MARK_TTS_BACKEND", "").strip()
        if ev:
            print(
                f"[TTS] MARK_TTS_BACKEND={ev!r} is set — it overrides api_keys.json. "
                "Unset it to use the UI / file ``tts_backend: gemini`` setting."
            )
        return False
    key = get_gemini_api_key()
    if not key:
        print(
            "[TTS] Gemini voice selected but ``gemini_api_key`` is empty in api_keys.json."
        )
        return False
    utter = (text or "").strip()
    if not utter:
        return False
    try:
        from google import genai
        from google.genai import types
    except ImportError as ex:
        print(f"[TTS] google-genai import failed: {ex}")
        return False
    primary = get_gemini_tts_model()
    voice = get_gemini_live_voice_name()
    client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})

    models_to_try = [primary]
    if primary.strip() != _ALT_TTS_MODEL:
        models_to_try.append(_ALT_TTS_MODEL)

    response = None
    model_used = primary
    for i, model in enumerate(models_to_try):
        try:
            response = client.models.generate_content(
                model=model,
                contents=utter,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        ),
                    ),
                ),
            )
            model_used = model
            break
        except Exception as ex:
            if _is_gemini_tts_quota_error(ex) and i + 1 < len(models_to_try):
                print(
                    f"[TTS] Gemini TTS model {model!r} is rate-limited or over quota; "
                    f"retrying {_ALT_TTS_MODEL!r}…"
                )
                continue
            if _is_gemini_tts_quota_error(ex):
                print(
                    "[TTS] Gemini TTS: quota or rate limit (429 / RESOURCE_EXHAUSTED). "
                    "Free tier has low per-model limits — you will hear Windows SAPI until "
                    "quota resets or billing is enabled. "
                    "https://ai.google.dev/gemini-api/docs/rate-limits"
                )
            else:
                print(f"[TTS] Gemini TTS request failed: {ex}")
            return False

    if response is None:
        return False

    got = _gemini_extract_audio(response)
    if not got:
        cands = getattr(response, "candidates", None) or []
        if not cands:
            print("[TTS] Gemini TTS: empty candidates (prompt blocked or model error).")
        else:
            c0 = cands[0]
            fr = getattr(c0, "finish_reason", None)
            print(
                f"[TTS] Gemini TTS returned no audio parts "
                f"(finish_reason={fr!r}). Check model id ``{model_used}`` and API quotas."
            )
        return False
    data, mime = got
    try:
        if "wav" in mime or data[:4] == b"RIFF":
            _play_wav_bytes_riff(data)
        else:
            # Default PCM from Gemini TTS docs: 24 kHz mono int16
            rate = 24000
            if "24000" in mime:
                rate = 24000
            elif "48000" in mime:
                rate = 48000
            _play_pcm_int16_mono(data, rate)
    except Exception as ex:
        print(f"[TTS] Gemini audio playback failed: {ex}")
        return False
    print(f"[TTS] Gemini TTS voice={voice!r} model={model_used!r}")
    return True


def speak_local_tts(text: str) -> None:
    """``pyttsx3`` only (Windows SAPI). Serialized with other TTS paths."""
    if not (text or "").strip():
        return
    with _tts_lock:
        _speak_pyttsx3_no_lock(text)


def speak_mark_tts(text: str) -> None:
    """
    Local Jarvis speech: **Gemini TTS** when ``tts_backend`` is ``gemini`` and a
    ``gemini_api_key`` is set; otherwise **pyttsx3**.
    """
    if not (text or "").strip():
        return
    global _tts_backend_hint_printed
    with _tts_lock:
        from mark_llm_settings import get_gemini_api_key, get_local_tts_backend

        if (
            not _tts_backend_hint_printed
            and get_local_tts_backend() != "gemini"
            and get_gemini_api_key()
        ):
            print(
                "[TTS] You have gemini_api_key but tts_backend is not \"gemini\" — "
                "replies use Windows SAPI (e.g. Zira). In the UI under VOICE OUTPUT "
                "(LOCAL), choose \"Gemini neural (uses API key)\", or set "
                "\"tts_backend\": \"gemini\" in config/api_keys.json."
            )
            _tts_backend_hint_printed = True
        if _try_gemini_tts(text):
            return
        if get_local_tts_backend() == "gemini":
            print(
                "[TTS] Gemini TTS failed or returned no usable audio — "
                "falling back to Windows SAPI (e.g. Zira). See [TTS] lines above."
            )
        _speak_pyttsx3_no_lock(text)
