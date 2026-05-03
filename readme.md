# 🤖 MARK XXXIX (39)
### The Ultimate Cross-Platform Personal AI Assistant — By FatihMakes

> 📺 **[Watch the full setup video on YouTube](https://youtu.be/ej1f5OE3SNQ?si=lCxDhJix9ungq1Ry)**

A real-time voice AI that can hear, see, understand, and control your computer — on any OS. Supporting Windows, macOS, and Linux. Local execution. Zero subscriptions. Engineered for total autonomy.

---

## ✨ Overview

MARK XXXIX represents the pinnacle of the Jarvis series, evolving into a more flexible and robust system. It bridges the gap between the operating system and human intent. Through natural dialogue, Mark 39 analyzes your screen, processes uploaded documents, and executes complex workflows with a brand-new, adaptive interface.

It's not just an assistant — it's an extension of your digital life.

---

## 🚀 Capabilities

### Core Features
| Feature | Description |
|---|---|
| 🎙️ Real-time Voice | Ultra-low latency conversation in any language (Gemini Live) or **local push-to-talk** via Whisper + Ollama |
| 🖥️ System Control | Launch apps, manage files, execute terminal commands |
| 🧩 Autonomous Tasks | High-level planning for complex, multi-step goals |
| 👁️ Visual Awareness | Real-time screen processing and webcam vision (Gemini Live or **local Ollama vision**) |
| 🧠 Persistent Memory | Deeply remembers your projects, preferences, and personal context |
| ⌨️ Hybrid Input | Type commands, upload files, or use **hold-to-talk (PTT)** in local Ollama mode |

---

## 🆕 What's New in XXXIX

- 📂 **Advanced File Handling** — New support for direct file uploads. Drop PDFs, source code, or images into the assistant to have them analyzed, summarized, or edited instantly.
- 🎨 **Adaptive & Flexible UI** — A complete overhaul of the interface. The new UI is fully resizable and responsive, featuring transparency controls and customizable layouts to fit your workspace perfectly.
- 🐧🍎 **Refined Cross-Platform Stability** — Major fixes for macOS and Linux compatibility. Core system actions are now more consistent across all three major operating systems.
- ⚡ **Optimized Core Engine** — Significant performance boost in tool-calling logic and response generation, resulting in a 40% faster interaction speed.
- 🦙 **Local Ollama path (enhanced)** — Run without a Gemini API key for **chat**: **Ollama `/api/chat`** for tools + chat, **faster-whisper** for PTT speech-to-text, **separate vision model** for screen/camera and `screen_find`. Spoken replies can use **Windows SAPI (`pyttsx3`)** or **Gemini neural TTS** (same `google.genai` stack as the rest of the project; needs `gemini_api_key` only for speech while chat stays local).
- 🔊 **VOICE OUTPUT (LOCAL)** — In Ollama mode the right panel lets you pick **Windows (SAPI)** vs **Gemini neural**, choose a **prebuilt Gemini voice** (Charon, Kore, …), and persists `tts_backend` / `gemini_live_voice` to `config/api_keys.json`. Picking a Gemini voice auto-selects neural TTS when a key is present. **`mark_tts.py`** handles synthesis, **base64-safe** audio handling, and **429 / quota** fallbacks (retries an alternate TTS model before falling back to SAPI).

---

## ⚡ Quick Start

```bash
git clone https://github.com/FatihMakes/Mark-XXXIX.git
cd Mark-XXXIX
pip install -r requirements.txt
playwright install
python main.py
```

> ⚠️ **Installation Note:** To keep the repository lightweight, some OS-specific dependencies are not bundled in `requirements.txt`. If you run into a `ModuleNotFoundError`, simply install the missing package via `pip install <module_name>` for your specific system.

> 💡 **Tip:** If you use a global Python environment that also has **TensorFlow**, `pip` may warn about **protobuf** versions. A **dedicated venv** for Mark (`python -m venv .venv` then activate and reinstall) avoids clashes.

---

## 📋 Requirements

| Requirement | Details |
|---|---|
| **OS** | Windows 10/11, macOS, or Linux |
| **Python** | 3.11 or 3.12 |
| **Microphone** | Required for Gemini Live; optional for **local Ollama** unless you use **PTT** (then needed for capture) |
| **API Key** | Optional if you use **local Ollama** instead of Gemini (see below) |
| **[Ollama](https://ollama.com)** | For local mode: daemon on **`http://127.0.0.1:11434`** by default; pull at least one **chat** model and one **vision** model (e.g. `llava`) for full screen features |

---

## 🦙 Local Ollama (no Gemini API key)

This fork adds a complete **local stack** alongside the original Gemini Live path.

### How it fits together

| Piece | Role |
|------|------|
| **Ollama chat model** (dropdown / `ollama_model`) | Reasoning, tool calls, replies via `POST /api/chat` |
| **Ollama vision model** (`MARK_OLLAMA_VISION_MODEL` or `ollama_vision_model`, default **`llava`**) | `screen_process`, webcam/screen questions, and **`screen_find`** — independent of the chat tag |
| **faster-whisper** (PTT) | Converts microphone audio to text; **first run** may download Whisper weights (separate from `ollama pull`) |
| **Voice output** (`tts_backend`) | **`pyttsx3`** — Windows SAPI (optional `tts_voice_substring` / `MARK_TTS_VOICE` to pick e.g. Zira). **`gemini`** — `generate_content` on a **TTS model** with a **prebuilt voice**; audio plays via **sounddevice** (same PortAudio path as `main.py`). |
| **`mark_tts.py`** | Chooses Gemini vs SAPI, calls Gemini TTS, normalizes API audio blobs, and logs quota / fallback reasons to the console. |

**Audio I/O:** The project uses **`sounddevice`** (PortAudio), not PyAudio.

### Setup checklist

1. Start **`ollama serve`** (or the Ollama app).
2. **`ollama pull`** a chat model (e.g. **`llama3.1:8b`**, `qwen2.5-coder:7b`) and a vision model (e.g. **`llava`**).
3. **`pip install -r requirements.txt`** and **`playwright install`**.
4. Run **`python main.py`** and choose **CONNECT LOCAL OLLAMA** in the setup overlay (or set `MARK_LLM_PROVIDER=ollama` and configure `config/api_keys.json`). **`config/api_keys.json` is gitignored** — do not commit your keys.

### UI in local mode

- **OLLAMA MODEL** — Populated from **`GET /api/tags`**. Changing it saves to `config/api_keys.json`. Use **↻** after `ollama pull`.
- **VOICE OUTPUT (LOCAL)** — First dropdown: **Windows (SAPI / pyttsx3)** or **Gemini neural (uses API key)** → saves **`tts_backend`**. Second dropdown: **Gemini voice** (prebuilt names) → saves **`gemini_live_voice`**. You need **`gemini_api_key`** in `api_keys.json` for neural output; chat still uses Ollama. Activity log shows e.g. `SYS: Voice output → gemini, Gemini voice=Kore`. If **`MARK_TTS_BACKEND`** is set in the environment, it **overrides** the file — the UI hint mentions this.
- **LOCAL VOICE (PTT)** — Appears when the Ollama backend is online: **hold** the button, speak, **release** to transcribe and send text to the **chat** model.

### Hybrid voice: Ollama chat + Gemini TTS (recommended flow)

1. Use **local Ollama** setup so the **OLLAMA MODEL** and voice panels are visible.
2. Add a **`gemini_api_key`** to `config/api_keys.json` (same key as full Gemini mode; file stays gitignored).
3. Under **VOICE OUTPUT (LOCAL)**, choose **Gemini neural (uses API key)** — sets **`tts_backend`** to **`gemini`**.
4. Pick a **Gemini voice** (e.g. Kore, Charon). That sets **`gemini_live_voice`**.
5. Restart **`python main.py`** after hand-editing JSON if you bypass the UI.

**Config keys (local speech):** `tts_backend` (`gemini` | `pyttsx3`), `gemini_live_voice`, optional `gemini_tts_model` (default in code is **`gemini-3.1-flash-tts-preview`**; override with **`MARK_GEMINI_TTS_MODEL`**), optional `tts_voice_substring` for SAPI when on Windows mode.

### Gemini TTS quotas and billing

Each spoken line uses the **Gemini API** (`models.generateContent` on a **TTS** model). **Free tier** limits are easy to hit (e.g. **429 RESOURCE_EXHAUSTED**); the app then **falls back to Windows SAPI** — often **Zira** if `tts_voice_substring` matches Zira — so it can *sound* like the UI failed when it is really **quota**. Watch the **terminal** for `[TTS]` lines. See [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) and [speech generation](https://ai.google.dev/gemini-api/docs/speech-generation).

### Ollama URL vs other proxies

Mark expects the **native Ollama HTTP API** at the configured base URL:

- **`GET {base}/api/tags`**
- **`POST {base}/api/chat`** with Ollama’s JSON body (and optional `images` on user messages for vision)

Default base is **`http://127.0.0.1:11434`**. A service on another port (e.g. **8082**) is **not** a port conflict with Ollama, but Mark will **only** work pointed at that host if it implements the **same** paths and payloads as Ollama (not only a health JSON at `/`). OpenAI-style gateways need different client code.

### Environment overrides

Optional; Aletheon-style names are supported where noted.

| Variable | Purpose |
|---|---|
| `MARK_LLM_PROVIDER` | `ollama` to force local mode, or `gemini` for cloud. |
| `MARK_OLLAMA_URL` | Ollama base URL (default `http://127.0.0.1:11434`). |
| `MARK_OLLAMA_MODEL` | Fixed chat model tag; overrides config and **disables** the UI dropdown when set. |
| `ALETHEON_LLM_ASSIST_OLLAMA_URL` | Same as `MARK_OLLAMA_URL` if the latter is unset. |
| `ALETHEON_LLM_ASSIST_OLLAMA_MODEL` | Same as `MARK_OLLAMA_MODEL` if the latter is unset. |
| `MARK_OLLAMA_VISION_MODEL` | Vision tag for screen/camera / `screen_find`; default **`llava`** if unset. |
| `MARK_DISABLE_TTS` | `1` / `true` to skip spoken replies in local mode. |
| `MARK_TTS_BACKEND` | `gemini` or `pyttsx3` — **overrides** `tts_backend` in `api_keys.json` when set. |
| `MARK_GEMINI_TTS_MODEL` | TTS model id for Gemini speech (overrides `gemini_tts_model` in JSON). |
| `MARK_GEMINI_VOICE` / `GEMINI_LIVE_VOICE` | Prebuilt Gemini voice name (overrides `gemini_live_voice` in JSON). |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Same keys as cloud Gemini; used for **Gemini TTS** in hybrid mode. |
| `MARK_WHISPER_SIZE` | Whisper size: `tiny` … `large-v3` (default `small`). |
| `MARK_WHISPER_DEVICE` | `cpu` or `cuda` for faster-whisper (default `cpu`). |
| `MARK_WHISPER_COMPUTE` | Override compute type (e.g. for GPU). |
| `MARK_WHISPER_LANGUAGE` | Whisper language code (default `en`). |

Gemini **Live** (mic streamed to the model, native audio) still requires the Gemini path. **Local mode** uses typed input, **PTT → Whisper → Ollama**, tools, and **Ollama vision** for screen tools as above.

---

## 🧩 Local stack: files touched (for contributors)

| Area | Files |
|------|--------|
| LLM routing / Ollama HTTP | `mark_llm_settings.py` |
| Local assistant loop + PTT queue | `jarvis_ollama.py`, `main.py` |
| Tool runner (Ollama branches) | `jarvis_tool_runner.py` |
| Speech-to-text | `mark_voice.py` |
| Local TTS (Gemini + SAPI) | `mark_tts.py` |
| UI (Ollama model, voice output, PTT) | `ui.py` |
| Screen vision + Gemini path | `actions/screen_processor.py` |
| `screen_find` + desktop tools | `actions/computer_control.py` |
| Other tools / planner using LLM config | `actions/code_helper.py`, `actions/dev_agent.py`, `actions/youtube_video.py`, `agent/planner.py`, `memory/config_manager.py` |
| Dependencies | `requirements.txt` (`faster-whisper`, etc.) |

---

## ⚠️ License

Personal and non-commercial use only.
Licensed under **[Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**.

---

## 👤 Connect with the Creator

Engineered by a developer building a real-world JARVIS-style assistant.
⭐ **Star the repository to support the journey to Mark 100.**

| Platform | Link |
|---|---|
| YouTube | [@FatihMakes](https://www.youtube.com/@FatihMakes) |
| Instagram | [@fatihmakes](https://www.instagram.com/fatihmakes) |
