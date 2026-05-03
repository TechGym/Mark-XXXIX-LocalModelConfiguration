"""Local Ollama backend: text commands + tool calling + offline TTS (no Gemini API)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Union

import mark_voice
import requests
from mark_llm_settings import get_ollama_model, get_ollama_url, ollama_chat
from mark_tts import speak_mark_tts

from jarvis_tool_runner import (
    ollama_tools_from_gemini_declarations,
    parse_tool_arguments,
    run_jarvis_tool,
    synthetic_tool_calls_from_text,
)
from memory.memory_manager import format_memory_for_prompt, load_memory


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


PROMPT_PATH = _base_dir() / "core" / "prompt.txt"


def _tts_say(text: str) -> None:
    if os.environ.get("MARK_DISABLE_TTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    if not (text or "").strip():
        return
    try:
        speak_mark_tts(text)
    except Exception as e:
        print(f"[TTS] speak failed: {e}")


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return (
            "You are JARVIS, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )


QueueItem = Union[str, tuple[str, bytes, int]]


class JarvisOllama:
    """
    Ollama-powered assistant loop (Aletheon-style HTTP to ``/api/chat``).

    Use the text field, file uploads, or **hold-to-talk** (PTT) in the UI; PTT audio
    is transcribed with faster-whisper then sent to Ollama. Spoken replies use
    **Gemini TTS** (if ``tts_backend`` is ``gemini`` and ``gemini_api_key`` is set)
    or **pyttsx3** / SAPI otherwise.
    """

    def __init__(self, ui, tool_declarations: list[dict]) -> None:
        self.ui = ui
        self._tool_declarations = tool_declarations
        self._loop: asyncio.AbstractEventLoop | None = None
        self._user_queue: asyncio.Queue[QueueItem] | None = None
        self._ollama_tools = ollama_tools_from_gemini_declarations(tool_declarations)
        # Set False after Ollama returns HTTP 400 with tools (e.g. vision-only chat tag).
        self._ollama_tools_enabled = True

    def _build_system_instruction(self) -> str:
        from datetime import datetime

        memory = load_memory()
        mem_str = format_memory_for_prompt(memory)
        sys_prompt = _load_system_prompt()
        now = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {time_str}\n"
            f"Use this to calculate exact times for reminders.\n\n"
        )
        parts = [time_ctx]
        if mem_str:
            parts.append(mem_str)
        parts.append(sys_prompt)
        parts.append(
            "\n[LOCAL MODE]\n"
            "You are running on a local Ollama model. Use tools whenever the user "
            "asks for an action. After tools complete, reply briefly in natural language.\n"
            "If the user asks you to greet or pass a short message to someone they name "
            "(e.g. a family member), do so in character; they are not asking you to "
            "look up external contact records or private data."
        )
        return "\n".join(parts)

    def speak(self, text: str) -> None:
        if self._loop and text:
            asyncio.run_coroutine_threadsafe(
                self._speak_async(text),
                self._loop,
            )

    def speak_error(self, tool_name: str, error: str) -> None:
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Sir, {tool_name} encountered an error. {short}")

    async def _speak_async(self, text: str) -> None:
        self.ui.set_state("SPEAKING")
        try:
            await asyncio.to_thread(_tts_say, text)
        finally:
            if not self.ui.muted:
                self.ui.set_state("LISTENING")

    def _wire_text_input(self) -> None:
        assert self._loop and self._user_queue

        def _enqueue(text: str) -> None:
            if not self._loop or not self._user_queue:
                return
            # Thread-safe: schedule asyncio.Queue.put (not put_nowait via call_soon_threadsafe).
            fut = asyncio.run_coroutine_threadsafe(
                self._user_queue.put(text),
                self._loop,
            )

            def _log_put_err(f) -> None:
                try:
                    exc = f.exception()
                except Exception:
                    return
                if exc is not None:
                    print(f"[JARVIS] ⚠️ queue put failed: {exc}")

            fut.add_done_callback(_log_put_err)

        self.ui.on_text_command = _enqueue

    def feed_pcm(self, pcm: bytes, sample_rate: int = 16000) -> None:
        """Queue raw int16 mono PCM from the UI PTT (thread-safe)."""
        if not self._loop or not self._user_queue:
            return
        fut = asyncio.run_coroutine_threadsafe(
            self._user_queue.put(("pcm", pcm, sample_rate)),
            self._loop,
        )

        def _log_err(f) -> None:
            try:
                exc = f.exception()
            except Exception:
                return
            if exc is not None:
                print(f"[JARVIS] ⚠️ PTT queue put failed: {exc}")

        fut.add_done_callback(_log_err)

    async def _ollama_chat_round(self, messages: list[dict]) -> dict:
        """
        POST /api/chat.

        On HTTP 400 or 500 while tools are enabled, retry once without tools and
        disable tools for the rest of the session (registry may reject tools, or
        the runner may crash on the tool payload / VRAM).
        """
        tools = self._ollama_tools if self._ollama_tools_enabled else None

        def _call(ts: list[dict] | None) -> dict:
            return ollama_chat(messages, tools=ts)

        try:
            return await asyncio.to_thread(_call, tools)
        except requests.HTTPError as e:
            resp = e.response
            hint = (resp.text or "").strip()[:400] if resp is not None else ""
            code = resp.status_code if resp is not None else None
            if code in (400, 500) and tools:
                if code == 400:
                    self.ui.write_log(
                        "SYS: Ollama HTTP 400 with tools — retrying without tools. "
                        "Pick a model that supports tools (e.g. llama3.1:8b) in OLLAMA MODEL."
                        + (f" Detail: {hint}" if hint else "")
                    )
                else:
                    self.ui.write_log(
                        "SYS: Ollama HTTP 500 with tools — retrying without tools "
                        "(often VRAM or runner error with this model + tool list)."
                        + (f" Detail: {hint}" if hint else "")
                    )
                self._ollama_tools_enabled = False
                return await asyncio.to_thread(_call, None)
            if hint:
                self.ui.write_log(f"ERR: Ollama HTTP {code}: {hint}")
            raise

    async def _run_one_exchange(self, user_text: str) -> None:
        assert self._user_queue is not None
        self.ui.set_state("THINKING")
        sys_instr = self._build_system_instruction()
        messages: list[dict] = [
            {"role": "system", "content": sys_instr},
            {"role": "user", "content": user_text},
        ]

        for _round in range(12):
            data = await self._ollama_chat_round(messages)
            msg = data.get("message") or {}
            content = (msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls and content:
                valid = {
                    (t.get("function") or {}).get("name") or ""
                    for t in (self._ollama_tools or [])
                    if isinstance(t, dict)
                }
                valid.discard("")
                synthetic = synthetic_tool_calls_from_text(
                    content, valid_names=valid
                )
                if synthetic:
                    self.ui.write_log(
                        "SYS: Model returned tool syntax as plain text — executing it once. "
                        "Prefer models that emit native tool_calls for reliability."
                    )
                    tool_calls = synthetic
                    content = ""

            assistant_msg: dict = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if tool_calls:
                loop = asyncio.get_running_loop()
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    tname = fn.get("name") or ""
                    args = parse_tool_arguments(fn.get("arguments"))
                    print(f"[JARVIS] 📞 {tname}")
                    out = await run_jarvis_tool(
                        tname,
                        args,
                        ui=self.ui,
                        speak=self.speak,
                        speak_error=self.speak_error,
                        loop=loop,
                    )
                    tool_body = json.dumps(out, ensure_ascii=False)
                    tool_entry: dict = {
                        "role": "tool",
                        "content": tool_body,
                    }
                    if tname:
                        tool_entry["name"] = tname
                    tid = tc.get("id")
                    if tid:
                        tool_entry["tool_call_id"] = str(tid)
                    messages.append(tool_entry)
                continue

            if content:
                self.ui.write_log(f"Jarvis: {content}")
                await self._speak_async(content)
            return

        self.ui.write_log("Jarvis: (no response after tool rounds)")
        await self._speak_async("Sir, I hit an internal reasoning limit on that request.")

    async def run(self) -> None:
        self._loop = asyncio.get_event_loop()
        self._user_queue = asyncio.Queue()
        self._wire_text_input()

        win = getattr(self.ui, "_win", None)
        if win is not None and hasattr(win, "ollama_attached"):
            win.ollama_attached.emit(self)

        print(
            f"[JARVIS] 🦙 Local Ollama mode — {get_ollama_url()} model={get_ollama_model()}"
        )
        print("[JARVIS] ℹ️  Type commands, upload files, or use hold-to-talk (PTT) in the UI.")

        while True:
            try:
                self.ui.set_state("LISTENING")
                self.ui.write_log("SYS: JARVIS online (local Ollama).")
                while True:
                    item: Any = await self._user_queue.get()
                    try:
                        if (
                            isinstance(item, tuple)
                            and len(item) == 3
                            and item[0] == "pcm"
                        ):
                            _, pcm, sr = item
                            if not pcm or len(pcm) < 3200:
                                self.ui.write_log("SYS: PTT clip too short — ignored.")
                                continue
                            self.ui.set_state("THINKING")
                            self.ui.write_log("SYS: Transcribing (Whisper)…")
                            text = await asyncio.to_thread(
                                mark_voice.transcribe_pcm_int16,
                                pcm,
                                int(sr),
                            )
                            text = (text or "").strip()
                            if not text:
                                self.ui.write_log("SYS: No speech detected.")
                                if not self.ui.muted:
                                    self.ui.set_state("LISTENING")
                                continue
                            self.ui.write_log(f"You (PTT): {text}")
                            await self._run_one_exchange(text)
                            continue

                        user_text = str(item or "").strip()
                        if not user_text:
                            continue
                        await self._run_one_exchange(user_text)
                    except Exception as ex:
                        short = str(ex).strip()[:400]
                        self.ui.write_log(f"ERR: {short}")
                        traceback.print_exc()
                        if not self.ui.muted:
                            self.ui.set_state("LISTENING")
            except Exception as e:
                print(f"[JARVIS] ⚠️ {e}")
                traceback.print_exc()
            self.ui.set_state("THINKING")
            print("[JARVIS] 🔄 Reconnecting local loop in 3s...")
            await asyncio.sleep(3)
