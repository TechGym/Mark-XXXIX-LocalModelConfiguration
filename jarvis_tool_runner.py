"""Shared async tool execution for Gemini Live and local Ollama backends."""

from __future__ import annotations

import asyncio
import json
import threading
import traceback
from typing import Any, Callable

from memory.memory_manager import update_memory

from actions.browser_control import browser_control
from actions.code_helper import code_helper
from actions.computer_control import computer_control
from actions.computer_settings import computer_settings
from actions.desktop import desktop_control
from actions.dev_agent import dev_agent
from actions.file_controller import file_controller
from actions.file_processor import file_processor
from actions.flight_finder import flight_finder
from actions.game_updater import game_updater
from actions.open_app import open_app
from actions.reminder import reminder
from actions.screen_processor import screen_process
from actions.send_message import send_message
from actions.weather_report import weather_action
from actions.web_search import web_search as web_search_action
from actions.youtube_video import youtube_video


SpeakFn = Callable[[str], None]
SpeakErrFn = Callable[[str, str], None]


async def run_jarvis_tool(
    name: str,
    args: dict,
    *,
    ui,
    speak: SpeakFn,
    speak_error: SpeakErrFn,
    loop: asyncio.AbstractEventLoop,
) -> dict[str, Any]:
    """
    Execute one JARVIS tool by name.

    Returns a dict: ``{"result": str|...}`` or ``{"result": "ok", "silent": True}`` for save_memory.
    """
    print(f"[JARVIS] 🔧 {name}  {args}")
    ui.set_state("THINKING")

    if name == "save_memory":
        category = args.get("category", "notes")
        key = args.get("key", "")
        value = args.get("value", "")
        if key and value:
            update_memory({category: {key: {"value": value}}})
            print(f"[Memory] 💾 save_memory: {category}/{key} = {value}")
        if not ui.muted:
            ui.set_state("LISTENING")
        return {"result": "ok", "silent": True}

    result: str = "Done."

    try:
        if name == "open_app":
            r = await loop.run_in_executor(
                None, lambda: open_app(parameters=args, response=None, player=ui)
            )
            result = r or f"Opened {args.get('app_name')}."

        elif name == "weather_report":
            r = await loop.run_in_executor(
                None, lambda: weather_action(parameters=args, player=ui)
            )
            result = r or "Weather delivered."

        elif name == "browser_control":
            r = await loop.run_in_executor(
                None, lambda: browser_control(parameters=args, player=ui)
            )
            result = r or "Done."

        elif name == "file_controller":
            r = await loop.run_in_executor(
                None, lambda: file_controller(parameters=args, player=ui)
            )
            result = r or "Done."

        elif name == "send_message":
            r = await loop.run_in_executor(
                None,
                lambda: send_message(
                    parameters=args, response=None, player=ui, session_memory=None
                ),
            )
            result = r or f"Message sent to {args.get('receiver')}."

        elif name == "reminder":
            r = await loop.run_in_executor(
                None, lambda: reminder(parameters=args, response=None, player=ui)
            )
            result = r or "Reminder set."

        elif name == "youtube_video":
            r = await loop.run_in_executor(
                None, lambda: youtube_video(parameters=args, response=None, player=ui)
            )
            result = r or "Done."

        elif name == "screen_process":
            threading.Thread(
                target=screen_process,
                kwargs={
                    "parameters": args,
                    "response": None,
                    "player": ui,
                    "session_memory": None,
                },
                daemon=True,
            ).start()
            result = (
                "Vision module activated. Stay completely silent — "
                "vision module will speak directly."
            )

        elif name == "computer_settings":
            r = await loop.run_in_executor(
                None,
                lambda: computer_settings(parameters=args, response=None, player=ui),
            )
            result = r or "Done."

        elif name == "desktop_control":
            r = await loop.run_in_executor(
                None, lambda: desktop_control(parameters=args, player=ui)
            )
            result = r or "Done."

        elif name == "code_helper":
            r = await loop.run_in_executor(
                None, lambda: code_helper(parameters=args, player=ui, speak=speak)
            )
            result = r or "Done."

        elif name == "dev_agent":
            r = await loop.run_in_executor(
                None, lambda: dev_agent(parameters=args, player=ui, speak=speak)
            )
            result = r or "Done."

        elif name == "agent_task":
            from agent.task_queue import TaskPriority, get_queue

            priority_map = {
                "low": TaskPriority.LOW,
                "normal": TaskPriority.NORMAL,
                "high": TaskPriority.HIGH,
            }
            priority = priority_map.get(
                args.get("priority", "normal").lower(), TaskPriority.NORMAL
            )
            task_id = get_queue().submit(
                goal=args.get("goal", ""), priority=priority, speak=speak
            )
            result = f"Task started (ID: {task_id})."

        elif name == "web_search":
            r = await loop.run_in_executor(
                None, lambda: web_search_action(parameters=args, player=ui)
            )
            result = r or "Done."

        elif name == "file_processor":
            if not args.get("file_path") and ui.current_file:
                args = {**args, "file_path": ui.current_file}
            r = await loop.run_in_executor(
                None,
                lambda: file_processor(parameters=args, player=ui, speak=speak),
            )
            result = r or "Done."

        elif name == "computer_control":
            r = await loop.run_in_executor(
                None, lambda: computer_control(parameters=args, player=ui)
            )
            result = r or "Done."

        elif name == "game_updater":
            r = await loop.run_in_executor(
                None, lambda: game_updater(parameters=args, player=ui, speak=speak)
            )
            result = r or "Done."

        elif name == "flight_finder":
            r = await loop.run_in_executor(
                None, lambda: flight_finder(parameters=args, player=ui)
            )
            result = r or "Done."

        elif name == "shutdown_jarvis":
            ui.write_log("SYS: Shutdown requested.")
            speak("Goodbye, sir.")

            def _shutdown() -> None:
                import os
                import time

                time.sleep(1)
                os._exit(0)

            threading.Thread(target=_shutdown, daemon=True).start()

        else:
            result = f"Unknown tool: {name}"

    except Exception as e:
        result = f"Tool '{name}' failed: {e}"
        traceback.print_exc()
        speak_error(name, str(e))

    if not ui.muted:
        ui.set_state("LISTENING")

    print(f"[JARVIS] 📤 {name} → {str(result)[:80]}")
    return {"result": result}


def ollama_tools_from_gemini_declarations(
    declarations: list[dict],
) -> list[dict]:
    """Convert Gemini-style function_declarations to Ollama/OpenAI-style tools."""

    def norm_schema(node: object) -> object:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == "type" and isinstance(v, str):
                    t = v.upper()
                    mapped = {
                        "OBJECT": "object",
                        "STRING": "string",
                        "INTEGER": "integer",
                        "NUMBER": "number",
                        "BOOLEAN": "boolean",
                        "ARRAY": "array",
                    }.get(t, v.lower() if t.isupper() and len(t) > 1 else v)
                    out[k] = mapped
                else:
                    out[k] = norm_schema(v)
            return out
        if isinstance(node, list):
            return [norm_schema(x) for x in node]
        return node

    tools: list[dict] = []
    for decl in declarations:
        params = decl.get("parameters") or {"type": "object", "properties": {}}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": decl["name"],
                    "description": decl.get("description", ""),
                    "parameters": norm_schema(params),
                },
            }
        )
    return tools


def parse_tool_arguments(raw: object) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}
