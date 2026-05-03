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
    return (
        os.environ.get("MARK_OLLAMA_URL", "").strip()
        or os.environ.get("ALETHEON_LLM_ASSIST_OLLAMA_URL", "").strip()
        or (_load_config().get("ollama_url") or "").strip()
        or "http://127.0.0.1:11434"
    ).rstrip("/")


def get_ollama_model() -> str:
    return (
        os.environ.get("MARK_OLLAMA_MODEL", "").strip()
        or os.environ.get("ALETHEON_LLM_ASSIST_OLLAMA_MODEL", "").strip()
        or (_load_config().get("ollama_model") or "").strip()
        or "dolphin-llama3:8b"
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
