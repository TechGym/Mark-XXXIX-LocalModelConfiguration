"""LLM backend selection: Gemini (cloud) or Ollama (local), aligned with Aletheon env conventions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


API_CONFIG_PATH = get_base_dir() / "config" / "api_keys.json"


def _load_config() -> dict:
    if not API_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _normalize_ollama_base(url: str) -> str:
    """
    Ollama native calls use ``{base}/api/chat``. If ``ollama_url`` was pasted with
    ``/api`` or OpenAI-style ``/v1`` suffixes, strip them so we do not POST to
    the wrong path (often surfaces as HTTP 405).
    """
    u = (url or "").strip().rstrip("/")
    if not u:
        return "http://127.0.0.1:11434"
    lower = u.lower()
    for suf in (
        "/v1/chat/completions",
        "/v1/chat",
        "/api/chat",
        "/api/generate",
        "/api/tags",
        "/api",
        "/v1",
    ):
        if lower.endswith(suf):
            u = u[: len(u) - len(suf)].rstrip("/")
            lower = u.lower()
    return u or "http://127.0.0.1:11434"


def is_ollama_mode() -> bool:
    """True when Mark should use local Ollama instead of Gemini."""
    env = os.environ.get("MARK_LLM_PROVIDER", "").strip().lower()
    if env == "ollama":
        return True
    if env == "gemini":
        return False
    data = _load_config()
    if (data.get("llm_provider") or "").strip().lower() == "ollama":
        return True
    return False


def get_ollama_url() -> str:
    raw = (
        os.environ.get("MARK_OLLAMA_URL", "").strip()
        or os.environ.get("ALETHEON_LLM_ASSIST_OLLAMA_URL", "").strip()
        or (_load_config().get("ollama_url") or "").strip()
        or "http://127.0.0.1:11434"
    )
    return _normalize_ollama_base(raw)


def get_ollama_model() -> str:
    return (
        os.environ.get("MARK_OLLAMA_MODEL", "").strip()
        or os.environ.get("ALETHEON_LLM_ASSIST_OLLAMA_MODEL", "").strip()
        or (_load_config().get("ollama_model") or "").strip()
        or "llama3.1:8b"
    )


def get_ollama_vision_model() -> str:
    """
    Vision-capable Ollama tag for screen/camera tools (separate from chat model).

    Set ``MARK_OLLAMA_VISION_MODEL`` or ``ollama_vision_model`` in ``api_keys.json``.
    Default ``llava`` — run ``ollama pull llava`` (or another vision tag) before use.
    """
    v = (
        os.environ.get("MARK_OLLAMA_VISION_MODEL", "").strip()
        or (_load_config().get("ollama_vision_model") or "").strip()
    )
    return v or "llava"


def get_gemini_live_voice_name() -> str:
    """
    Prebuilt Gemini voice id for **Live** sessions and for **local Gemini TTS**
    (when ``tts_backend`` is ``gemini``).

    Override with ``MARK_GEMINI_VOICE`` / ``GEMINI_LIVE_VOICE``, or
    ``gemini_live_voice`` / ``gemini_voice_name`` in ``config/api_keys.json``.
    Default ``Charon``. See Gemini speech docs for the full list.
    """
    v = (
        os.environ.get("MARK_GEMINI_VOICE", "").strip()
        or os.environ.get("GEMINI_LIVE_VOICE", "").strip()
        or (_load_config().get("gemini_live_voice") or "").strip()
        or (_load_config().get("gemini_voice_name") or "").strip()
    )
    return v or "Charon"


def get_local_tts_voice_substring() -> str | None:
    """
    Substring matched against Windows SAPI voice **display names** for ``pyttsx3``.

    First voice whose ``name`` contains this substring (case-insensitive) is used.
    Set ``MARK_TTS_VOICE``, or ``tts_voice_substring`` / ``local_tts_voice`` in
    ``config/api_keys.json``. Example substrings: ``David``, ``Zira``, ``Mark``.
    """
    v = (
        os.environ.get("MARK_TTS_VOICE", "").strip()
        or (_load_config().get("tts_voice_substring") or "").strip()
        or (_load_config().get("local_tts_voice") or "").strip()
    )
    return v or None


# Prebuilt Gemini TTS / Live voice names (see Gemini speech generation docs).
GEMINI_TTS_VOICE_NAMES: tuple[str, ...] = (
    "Achernar",
    "Achird",
    "Algenib",
    "Algieba",
    "Alnilam",
    "Aoede",
    "Autonoe",
    "Callirrhoe",
    "Charon",
    "Despina",
    "Enceladus",
    "Erinome",
    "Fenrir",
    "Gacrux",
    "Iapetus",
    "Kore",
    "Laomedeia",
    "Leda",
    "Orus",
    "Puck",
    "Pulcherrima",
    "Rasalgethi",
    "Sadachbia",
    "Sadaltager",
    "Schedar",
    "Sulafat",
    "Umbriel",
    "Vindemiatrix",
    "Zephyr",
    "Zubenelgenubi",
)


def list_gemini_tts_voice_names() -> list[str]:
    return list(GEMINI_TTS_VOICE_NAMES)


def get_gemini_api_key() -> str:
    """Same key as full Gemini mode: ``gemini_api_key`` in ``api_keys.json`` (or env)."""
    return (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
        or (_load_config().get("gemini_api_key") or "").strip()
    )


def get_local_tts_backend() -> str:
    """
    Local reply speech: ``pyttsx3`` (Windows SAPI) or ``gemini`` (Gemini TTS HTTP,
    uses ``gemini_api_key`` only for synthesis while chat stays on Ollama).

    ``MARK_TTS_BACKEND`` or ``tts_backend`` in ``api_keys.json``: ``gemini`` / ``pyttsx3``.
    """
    v = (
        os.environ.get("MARK_TTS_BACKEND", "").strip().lower()
        or (_load_config().get("tts_backend") or "").strip().lower()
    )
    if v in ("gemini", "google"):
        return "gemini"
    return "pyttsx3"


def get_gemini_tts_model() -> str:
    """
    Gemini TTS model id for ``generate_content`` (see Google speech docs).

    Default ``gemini-3.1-flash-tts-preview`` — older ``gemini-2.5-flash-preview-tts`` /
    ``gemini-2.5-flash-tts`` often share a **small free-tier quota**; 429s there fall
    back to Windows SAPI unless you set ``gemini_tts_model`` / ``MARK_GEMINI_TTS_MODEL``.
    """
    v = (
        os.environ.get("MARK_GEMINI_TTS_MODEL", "").strip()
        or (_load_config().get("gemini_tts_model") or "").strip()
    )
    return v or "gemini-3.1-flash-tts-preview"


def ollama_model_env_locked() -> bool:
    """When True, ``MARK_OLLAMA_MODEL`` / Aletheon env overrides config and the UI picker."""
    return bool(
        os.environ.get("MARK_OLLAMA_MODEL", "").strip()
        or os.environ.get("ALETHEON_LLM_ASSIST_OLLAMA_MODEL", "").strip()
    )


def list_ollama_models() -> list[str]:
    """
    Return model tags reported by the local Ollama daemon (``GET /api/tags``),
    same idea as Aletheon's ``list_available_models`` for the assist UI.
    """
    import requests

    base = get_ollama_url().rstrip("/")
    resp = requests.get(f"{base}/api/tags", timeout=20)
    resp.raise_for_status()
    body = resp.json()
    raw = body.get("models")
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for row in raw:
        if isinstance(row, dict):
            name = row.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return list(dict.fromkeys(names))


def ollama_chat(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    model: str | None = None,
    timeout: int = 600,
) -> dict:
    """POST /api/chat (non-streaming). Returns parsed JSON or raises on HTTP error."""
    import requests

    url = f"{get_ollama_url()}/api/chat"
    payload: dict = {
        "model": (model or "").strip() or get_ollama_model(),
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    resp = requests.post(url, json=payload, timeout=timeout)
    if not resp.ok:
        body = (resp.text or "").strip()[:1200]
        print(f"[OLLAMA] HTTP {resp.status_code} {url}\n{body}")
        if resp.status_code == 405:
            print(
                "[OLLAMA] HTTP 405: /api/chat expects POST from the app, not GET in a browser. "
                "If this persists, check ollama_url is only the daemon root "
                "(e.g. http://127.0.0.1:11434) and that Ollama is running on that port."
            )
        if resp.status_code >= 500:
            print(
                "[OLLAMA] HTTP 5xx: check `ollama ps`, GPU VRAM, and `ollama logs`; "
                "try `ollama pull` again or a smaller model if OOM."
            )
    resp.raise_for_status()
    return resp.json()


def ollama_generate_text(
    user_prompt: str,
    *,
    system_instruction: str | None = None,
    timeout: int = 300,
) -> str:
    """Single-turn text generation via /api/chat (no tools)."""
    messages: list[dict] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": user_prompt})
    data = ollama_chat(messages, tools=None, timeout=timeout)
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


def ollama_chat_with_image_reply(
    user_text: str,
    image_b64: str,
    *,
    system: str | None = None,
    model: str | None = None,
    tools: list[dict] | None = None,
    timeout: int = 180,
) -> str:
    """One user turn with a base64-encoded image (PNG/JPEG) for a vision model."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": user_text,
            "images": [image_b64],
        }
    )
    data = ollama_chat(
        messages,
        tools=tools,
        model=model or get_ollama_vision_model(),
        timeout=timeout,
    )
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()
