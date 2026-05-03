"""Shared async tool execution for Gemini Live and local Ollama backends."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import traceback
from typing import Any, Callable, Optional

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


def _balanced_json_slice(text: str, open_brace: int) -> Optional[str]:
    """Return the JSON object starting at ``open_brace``, or ``None`` if unbalanced."""
    if open_brace < 0 or open_brace >= len(text) or text[open_brace] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    quote_char = ""
    for j in range(open_brace, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote_char:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote_char = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace : j + 1]
    return None


def synthetic_tool_calls_from_text(
    content: str,
    *,
    valid_names: set[str],
) -> list[dict]:
    """
    Some local models print tool intent in ``message.content`` instead of Ollama
    ``tool_calls``. Supported whole-message shapes:

    - ``open_app({"app_name": "Notepad"})``
    - ``open_app {"app_name": "Notepad"}`` (space instead of parentheses)
    - ``{"name": "open_app", "arguments": {"app_name": "Notepad"}}``
    - A tool line buried after prose (each non-empty line is tried).
    - JSON starting mid-string (e.g. one line with text then ``{"name":...}``).
    """
    s = (content or "").strip()
    if not s or len(s) > 12_000:
        return []
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s).strip()

    def _one(name: str, args: dict) -> list[dict]:
        if name not in valid_names or not isinstance(args, dict):
            return []
        return [
            {
                "id": "from_model_text",
                "function": {
                    "name": name,
                    "arguments": args,
                },
            }
        ]

    def _from_parsed_tool_json(obj: object) -> list[dict]:
        """OpenAI-style ``{"name": "...", "arguments": {...}}`` (or a one-element list)."""
        if isinstance(obj, dict):
            fn_block = obj.get("function")
            if isinstance(fn_block, dict) and not isinstance(obj.get("name"), str):
                name = fn_block.get("name")
                raw_args = fn_block.get("arguments")
            else:
                name = obj.get("name") or obj.get("function") or obj.get("tool_name")
                if isinstance(name, dict):
                    name = name.get("name")
                raw_args = obj.get("arguments") or obj.get("parameters") or obj.get("args")
            if isinstance(name, str):
                if isinstance(raw_args, str):
                    args = parse_tool_arguments(raw_args)
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}
                if isinstance(args, dict) and args:
                    return _one(name, args)
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            return _from_parsed_tool_json(obj[0])
        return []

    # JSON: whole message is a single tool object
    if s.lstrip().startswith("{"):
        try:
            obj_whole = json.loads(s)
        except json.JSONDecodeError:
            obj_whole = None
        got = _from_parsed_tool_json(obj_whole)
        if got:
            return got

    _PAREN_TOOL = re.compile(
        r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\([ \t]*(\{[\s\S]*\})[ \t]*\)[ \t]*$"
    )

    def _try_line(line: str) -> list[dict]:
        line = (line or "").strip()
        if not line:
            return []
        if line.lstrip().startswith("{"):
            try:
                obj_line = json.loads(line)
            except json.JSONDecodeError:
                obj_line = None
            got = _from_parsed_tool_json(obj_line)
            if got:
                return got
        m_p = _PAREN_TOOL.match(line)
        if m_p:
            name, json_blob = m_p.group(1), m_p.group(2)
            if name in valid_names:
                args = parse_tool_arguments(json_blob)
                if isinstance(args, dict) and args:
                    got = _one(name, args)
                    if got:
                        return got
        for name in sorted(valid_names, key=len, reverse=True):
            if not line.startswith(name):
                continue
            n = len(name)
            if n < len(line) and line[n] not in " \t\n({":
                continue
            rest = line[n:].lstrip()
            if not rest.startswith("{"):
                continue
            brace = line.find("{", n)
            blob = _balanced_json_slice(line, brace)
            if not blob:
                continue
            args = parse_tool_arguments(blob)
            if isinstance(args, dict) and args:
                got = _one(name, args)
                if got:
                    return got
        return []

    candidates: list[str] = [s]
    for ln in s.splitlines():
        t = ln.strip()
        if t and t not in candidates:
            candidates.append(t)
    for cand in candidates:
        got = _try_line(cand)
        if got:
            return got
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        blob = _balanced_json_slice(s, i)
        if not blob or len(blob) < 12:
            continue
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        got = _from_parsed_tool_json(parsed)
        if got:
            return got
    return []
