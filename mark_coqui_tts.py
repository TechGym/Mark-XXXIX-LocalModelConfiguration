"""
Local Coqui / TechGym-TTS speech for Mark (optional ``tts_backend: coqui``).

Loads the ``TTS`` package from ``coqui_tts_repo_path`` (your clone root, containing a
``TTS/`` directory). On failure, ``mark_tts.speak_mark_tts`` falls back to Windows SAPI.

**Inference vs training:** Mark only **runs** pretrained models (``coqui_model_name`` or
checkpoint paths). You do **not** need the LJSpeech **dataset** zip, ``download_ljspeech.sh``,
or ``train_modelX.py`` / recipe bash scripts unless you are **training** a new checkpoint
yourself. For a first English voice, set ``coqui_model_name`` to a registry id such as
``tts_models/en/ljspeech/tacotron2-DDC`` — Coqui-TTS will **download model weights** on first load
(that is not the same as downloading the full LJSpeech corpus for training).

No Tortoise models: names containing ``tortoise`` are rejected here.

Diagnostics: when Coqui cannot speak, a checklist is printed so you can keep tuning
Coqui while Zira still plays the reply.
"""

from __future__ import annotations

# Pretrained English LJSpeech-style Tacotron2 in the upstream Coqui model zoo (inference).
COQUI_EXAMPLE_REGISTRY_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

import sys
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Sentinel: Coqui model init failed; do not call ``_build_engine()`` on every utterance
# (that was blocking ``asyncio.to_thread`` workers and made shutdown hang).
_COQUI_LOAD_FAILED = object()

_coqui_engine: Any = None
_coqui_init_lock = threading.Lock()
_coqui_last_emit_sig: str | None = None
_coqui_fail_fast_hint_printed = False


def _reject_tortoise(name: str) -> bool:
    return "tortoise" in (name or "").lower()


def _ensure_repo_on_path(repo: Path) -> bool:
    root = repo.resolve()
    if not (root / "TTS").is_dir():
        return False
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return True


def _coqui_checklist_lines(
    *,
    import_error: str | None,
    model_load_error: str | None,
    tortoise_blocked: str | None,
) -> list[str]:
    """Human-readable bullets from current settings (no engine required)."""
    from mark_llm_settings import (
        get_coqui_config_path,
        get_coqui_model_name,
        get_coqui_model_path,
        get_coqui_tts_repo_path,
        get_coqui_use_cuda,
        get_coqui_vocoder_config_path,
        get_coqui_vocoder_path,
    )

    lines: list[str] = []
    repo = (get_coqui_tts_repo_path() or "").strip()
    if not repo:
        lines.append(
            "``coqui_tts_repo_path``: empty — set your clone root **or** install Coqui into "
            "this same Python (``cd`` your TTS repo, ``pip install -e .``) so ``TTS`` imports "
            "without a path."
        )
        try:
            from mark_llm_settings import coqui_settings_debug_snippet

            lines.append(coqui_settings_debug_snippet())
        except Exception:
            pass
    else:
        root = Path(repo)
        if not root.is_dir():
            lines.append(f"``coqui_tts_repo_path``: not a directory — {repo!r}")
        elif not (root / "TTS").is_dir():
            lines.append(
                f"``coqui_tts_repo_path``: there is no ``TTS/`` subfolder under {root} "
                "(wrong folder or incomplete clone)."
            )
        else:
            lines.append(f"``coqui_tts_repo_path``: OK — {root}")

    model_name = (get_coqui_model_name() or "").strip()
    model_path = (get_coqui_model_path() or "").strip()
    config_path = (get_coqui_config_path() or "").strip()
    vocoder_path = (get_coqui_vocoder_path() or "").strip()
    vocoder_cfg = (get_coqui_vocoder_config_path() or "").strip()

    if model_name:
        lines.append(f"``coqui_model_name``: {model_name!r}")
    if model_path or config_path:
        lines.append(
            "Offline checkpoint: "
            f"``coqui_model_path``={'set' if model_path else 'missing'}, "
            f"``coqui_config_path``={'set' if config_path else 'missing'}."
        )
        if model_path and not Path(model_path).is_file():
            lines.append(f"``coqui_model_path`` is not a file: {model_path!r}")
        if config_path and not Path(config_path).is_file():
            lines.append(f"``coqui_config_path`` is not a file: {config_path!r}")
    if vocoder_path or vocoder_cfg:
        lines.append(
            "Vocoder: "
            f"path={'set' if vocoder_path else 'missing'}, "
            f"config={'set' if vocoder_cfg else 'missing'} "
            "(both required if you use custom vocoder paths)."
        )
    if not model_name and not (model_path and config_path):
        lines.append(
            "Model: set ``coqui_model_name`` (Coqui **registry** id, downloads weights on "
            "first run) or both ``coqui_model_path`` + ``coqui_config_path`` for an offline "
            "checkpoint you trained or copied."
        )
        lines.append(
            f"Example registry (inference only — no LJSpeech recipe / training): "
            f"``{COQUI_EXAMPLE_REGISTRY_MODEL}``"
        )

    lines.append(f"``coqui_use_cuda``: {bool(get_coqui_use_cuda())} (GPU if available).")

    if tortoise_blocked:
        lines.append(tortoise_blocked)
    if import_error:
        lines.append(f"Python import: {import_error}")
    if model_load_error:
        lines.append(f"Model constructor / checkpoint: {model_load_error}")
    return lines


def _coqui_emit_sig(
    lines: list[str],
    import_error: str | None,
    model_load_error: str | None,
) -> str:
    from mark_llm_settings import (
        get_coqui_config_path,
        get_coqui_model_name,
        get_coqui_model_path,
        get_coqui_tts_repo_path,
    )

    return "|".join(
        [
            (get_coqui_tts_repo_path() or "").strip(),
            (get_coqui_model_name() or "").strip(),
            (get_coqui_model_path() or "").strip(),
            (get_coqui_config_path() or "").strip(),
            import_error or "",
            (model_load_error or "")[:400],
        ]
    )


def _print_coqui_goal_checklist(
    *,
    import_error: str | None,
    model_load_error: str | None,
    tortoise_blocked: str | None,
) -> None:
    global _coqui_last_emit_sig
    lines = _coqui_checklist_lines(
        import_error=import_error,
        model_load_error=model_load_error,
        tortoise_blocked=tortoise_blocked,
    )
    sig = _coqui_emit_sig(lines, import_error, model_load_error)
    print(
        "[TTS] Coqui — local neural TTS is the goal here. "
        "Below is what still blocks Coqui (Windows SAPI may still play this reply):"
    )
    if sig == _coqui_last_emit_sig:
        print(
            "  • (same Coqui settings as the last logged attempt — change "
            "``api_keys.json`` / env, save, then try again.)"
        )
        return
    _coqui_last_emit_sig = sig
    for ln in lines:
        print(f"  • {ln}")


def reset_coqui_engine_cache() -> None:
    """
    Drop cached Coqui engine (e.g. after editing ``api_keys.json``).

    Clears both a loaded engine and a **cached init failure** so the next reply retries load.
    """
    global _coqui_engine, _coqui_last_emit_sig, _coqui_fail_fast_hint_printed
    with _coqui_init_lock:
        _coqui_engine = None
    _coqui_last_emit_sig = None
    _coqui_fail_fast_hint_printed = False


def _build_engine() -> Any | None:
    from mark_llm_settings import (
        get_coqui_config_path,
        get_coqui_model_name,
        get_coqui_model_path,
        get_coqui_use_cuda,
        get_coqui_vocoder_config_path,
        get_coqui_vocoder_path,
        get_coqui_tts_repo_path,
    )

    repo = (get_coqui_tts_repo_path() or "").strip()
    if repo:
        root = Path(repo)
        if not root.is_dir():
            _print_coqui_goal_checklist(
                import_error=None,
                model_load_error=None,
                tortoise_blocked=None,
            )
            return None
        if not _ensure_repo_on_path(root):
            _print_coqui_goal_checklist(
                import_error=None,
                model_load_error=None,
                tortoise_blocked=None,
            )
            return None
    import_err: str | None = None
    try:
        from TTS.api import TTS as CoquiTTS  # type: ignore[import-not-found]
    except Exception as ex:
        import_err = f"{type(ex).__name__}: {ex}"
        _print_coqui_goal_checklist(
            import_error=import_err,
            model_load_error=None,
            tortoise_blocked=None,
        )
        return None
    if not repo:
        print(
            "[TTS] Coqui: ``coqui_tts_repo_path`` is empty — using the ``TTS`` package from "
            "this Python environment (e.g. after ``pip install -e`` your clone here). "
            "You can still set ``coqui_tts_repo_path`` to pin a specific checkout."
        )

    model_path = (get_coqui_model_path() or "").strip()
    config_path = (get_coqui_config_path() or "").strip()
    vocoder_path = (get_coqui_vocoder_path() or "").strip()
    vocoder_cfg = (get_coqui_vocoder_config_path() or "").strip()
    model_name = (get_coqui_model_name() or "").strip()
    use_cuda = bool(get_coqui_use_cuda())
    if use_cuda:
        try:
            import torch

            if not torch.cuda.is_available():
                print(
                    "[TTS] Coqui: ``get_coqui_use_cuda()`` was true but ``torch.cuda.is_available()`` "
                    "is false — loading model on **CPU**."
                )
                use_cuda = False
        except Exception:
            use_cuda = False

    tortoise_msg: str | None = None
    if model_path and _reject_tortoise(model_path):
        tortoise_msg = "Tortoise checkpoints are disabled in Mark — use a non-Tortoise model."
    elif model_name and _reject_tortoise(model_name):
        tortoise_msg = (
            "Tortoise ``coqui_model_name`` values are disabled — pick another registry id."
        )
    if tortoise_msg:
        _print_coqui_goal_checklist(
            import_error=None,
            model_load_error=None,
            tortoise_blocked=tortoise_msg,
        )
        return None

    model_load_error: str | None = None
    try:
        if model_path and config_path:
            kw: dict[str, Any] = {
                "model_path": model_path,
                "config_path": config_path,
                "progress_bar": False,
                "gpu": use_cuda,
            }
            if vocoder_path and vocoder_cfg:
                kw["vocoder_path"] = vocoder_path
                kw["vocoder_config_path"] = vocoder_cfg
            return CoquiTTS(**kw)
        if model_name:
            return CoquiTTS(
                model_name=model_name,
                progress_bar=False,
                gpu=use_cuda,
            )
    except Exception as ex:
        model_load_error = f"{type(ex).__name__}: {ex}"

    if model_load_error:
        _print_coqui_goal_checklist(
            import_error=None,
            model_load_error=model_load_error,
            tortoise_blocked=None,
        )
        return None

    _print_coqui_goal_checklist(
        import_error=None,
        model_load_error=None,
        tortoise_blocked=None,
    )
    return None


def _get_engine() -> Any | None:
    global _coqui_engine, _coqui_fail_fast_hint_printed
    if _coqui_engine is _COQUI_LOAD_FAILED:
        return None
    if _coqui_engine is not None:
        return _coqui_engine
    with _coqui_init_lock:
        if _coqui_engine is _COQUI_LOAD_FAILED:
            return None
        if _coqui_engine is not None:
            return _coqui_engine
        eng = _build_engine()
        if eng is None:
            _coqui_engine = _COQUI_LOAD_FAILED
            if not _coqui_fail_fast_hint_printed:
                print(
                    "[TTS] Coqui: init failed — **will not retry** full load on every reply "
                    "(avoids freezing the app). Fix config / install, save in UI, or restart. "
                    "Fast SAPI fallback until then."
                )
                _coqui_fail_fast_hint_printed = True
            return None
        _coqui_engine = eng
        print("[TTS] Coqui: model loaded.")
        return eng


def _print_coqui_runtime_hint(head: str, detail: str) -> None:
    print(f"[TTS] Coqui — {head}: {detail}")
    print(
        "[TTS] Coqui — engine loaded but this utterance did not play via Coqui; "
        "see line above. Next backend (Gemini if enabled) or Windows SAPI follows."
    )


def try_speak_coqui(
    text: str,
    *,
    on_audio_start: Callable[[], None] | None,
    tts_lock: threading.Lock,
) -> bool:
    """
    Synthesize with local Coqui and play. Returns True on success; False → use SAPI.
    """
    utter = (text or "").strip()
    if not utter:
        return False

    eng = _get_engine()
    if eng is None:
        return False

    from mark_llm_settings import get_coqui_language, get_coqui_speaker

    speaker = (get_coqui_speaker() or "").strip() or None
    language = (get_coqui_language() or "").strip() or None

    try:
        kwargs: dict[str, Any] = {}
        if eng.is_multi_speaker:
            sp = speaker
            if not sp and getattr(eng, "speakers", None):
                sp = eng.speakers[0]
            if not sp:
                _print_coqui_runtime_hint(
                    "multi-speaker model needs ``coqui_speaker``",
                    "Set ``coqui_speaker`` in api_keys.json or pick a single-speaker model.",
                )
                return False
            kwargs["speaker"] = sp
        if eng.is_multi_lingual:
            lang = language
            if not lang and getattr(eng, "languages", None):
                lang = eng.languages[0]
            if not lang:
                _print_coqui_runtime_hint(
                    "multilingual model needs ``coqui_language``",
                    "Set ``coqui_language`` or use an English-only model.",
                )
                return False
            kwargs["language"] = lang

        wav = eng.tts(utter, **kwargs)
    except Exception as ex:
        _print_coqui_runtime_hint(
            "synthesis failed",
            f"{type(ex).__name__}: {ex}",
        )
        return False

    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        _print_coqui_runtime_hint("empty waveform", "Check model output or input text.")
        return False

    sr = int(getattr(eng.synthesizer, "output_sample_rate", None) or 22050)
    clipped = np.clip(arr, -1.0, 1.0)
    pcm_i16 = (clipped * 32767.0).astype(np.int16)
    pcm = pcm_i16.tobytes()

    try:
        import sounddevice as sd

        with tts_lock:
            if on_audio_start:
                on_audio_start()
            audio = np.frombuffer(pcm, dtype=np.int16)
            sd.play(audio, sr, blocking=True)
    except Exception as ex:
        _print_coqui_runtime_hint(
            "playback failed",
            f"{type(ex).__name__}: {ex}",
        )
        return False

    print("[TTS] Coqui: playback finished.")
    return True
