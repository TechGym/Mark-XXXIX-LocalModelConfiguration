import json
from datetime import datetime
from threading import Lock
from pathlib import Path
import sys


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR         = get_base_dir()
MEMORY_PATH      = BASE_DIR / "memory" / "long_term.json"
_lock            = Lock()
MAX_VALUE_LENGTH = 380
MEMORY_MAX_CHARS = 2200

def _empty_memory() -> dict:
    return {
        "identity":      {},
        "preferences":   {},
        "projects":      {},
        "relationships": {},
        "wishes":        {},
        "notes":         {},
    }

def load_memory() -> dict:
    if not MEMORY_PATH.exists():
        return _empty_memory()
    with _lock:
        try:
            data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                base = _empty_memory()
                for key in base:
                    if key not in data:
                        data[key] = {}
                return data
            return _empty_memory()
        except Exception as e:
            print(f"[Memory] ⚠️ Load error: {e}")
            return _empty_memory()

def _all_entries(memory: dict) -> list[tuple]:
    entries = []
    for cat, items in memory.items():
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            if isinstance(entry, dict) and "value" in entry:
                entries.append((cat, key, entry))
    return entries


def _trim_to_limit(memory: dict) -> dict:
    if len(json.dumps(memory, ensure_ascii=False)) <= MEMORY_MAX_CHARS:
        return memory
    entries = _all_entries(memory)
    entries.sort(key=lambda t: t[2].get("updated", "0000-00-00"))
    for cat, key, _ in entries:
        if len(json.dumps(memory, ensure_ascii=False)) <= MEMORY_MAX_CHARS:
            break
        del memory[cat][key]
        print(f"[Memory] 🗑️  Trimmed {cat}/{key}")
    return memory

def save_memory(memory: dict) -> None:
    if not isinstance(memory, dict):
        return
    memory = _trim_to_limit(memory)
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        MEMORY_PATH.write_text(
            json.dumps(memory, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _truncate_value(val: str) -> str:
    if isinstance(val, str) and len(val) > MAX_VALUE_LENGTH:
        return val[:MAX_VALUE_LENGTH].rstrip() + "…"
    return val


def _recursive_update(target: dict, updates: dict) -> bool:
    changed = False
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, dict) and "value" not in value:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
                changed = True
            if _recursive_update(target[key], value):
                changed = True
        else:
            new_val  = _truncate_value(str(value["value"] if isinstance(value, dict) else value))
            entry    = {"value": new_val, "updated": datetime.now().strftime("%Y-%m-%d")}
            existing = target.get(key, {})
            if not isinstance(existing, dict) or existing.get("value") != new_val:
                target[key] = entry
                changed = True
    return changed


def update_memory(memory_update: dict) -> dict:
    if not isinstance(memory_update, dict) or not memory_update:
        return load_memory()
    memory = load_memory()
    if _recursive_update(memory, memory_update):
        save_memory(memory)
        print(f"[Memory] 💾 Saved: {list(memory_update.keys())}")
    return memory


def _memory_entry_str(entry: object) -> str:
    if entry is None:
        return ""
    if isinstance(entry, dict):
        v = entry.get("value")
        if v is None:
            return ""
        return str(v).strip()
    return str(entry).strip()


def _registered_names_prompt(*, assistant: str, human_first: str) -> str:
    """
    Explicit A/B mapping. Avoid instruction text like "your name" — models often bind
    "you" to the wrong role when a human name appears nearby.
    """
    lines = [
        "[REGISTERED NAMES — fixed mapping; never swap A and B]",
        f"A) **Assistant** (the AI chatbot the human talks to): «{assistant}»",
    ]
    if human_first:
        lines.append(
            f"B) **Human user** (the living person at keyboard/mic): «{human_first}»"
        )
        lines.append("")
        lines.append("Route the user's question by who the question is about:")
        lines.append(
            f'• Question is about **the assistant** (second person toward the bot), e.g. '
            f'"What is your name?" meaning the bot → answer using **only** «{assistant}». '
            f'**Incorrect:** naming «{human_first}» as the bot.'
        )
        lines.append(
            f'• Question is about **the human themselves** (first person), e.g. '
            f'"What is my name?" → answer using **only** «{human_first}» for that person '
            f'(e.g. "Your name is {human_first}"). **Incorrect:** naming «{assistant}» here, '
            f'or saying the assistant\'s name is «{human_first}».'
        )
    else:
        lines.append("")
        lines.append(
            f'• User asks the bot\'s name → use **only** «{assistant}».'
        )
    lines.append("")
    lines.append(
        "Assistant **grammar**: In normal replies refer to yourself with **first person** "
        "(I, me, my) — not third person (she, her, herself). Do not write as if the assistant "
        f'were a separate person (avoid "{assistant} can …" and "She can …"; say "I can …"). '
        f'Use «{assistant}» when giving your name or when a name is explicitly required; otherwise prefer **I**.'
    )
    return "\n".join(lines) + "\n"


def assistant_persona_final_override(memory: dict | None) -> str:
    """
    Short system suffix so the chosen assistant display name wins over JARVIS
    branding in ``core/prompt.txt`` (small models often revert on follow-up questions).
    """
    if not memory or not isinstance(memory, dict):
        return ""
    identity = memory.get("identity")
    if not isinstance(identity, dict):
        return ""
    assistant = _memory_entry_str(identity.get("assistant_name"))
    if not assistant:
        return ""
    human = _memory_entry_str(identity.get("name"))
    body = _registered_names_prompt(assistant=assistant, human_first=human)
    if human:
        tail = (
            "\n[FINAL — SAME NAME RULES]\n"
            "If any other text conflicts, **A) is the assistant** and **B) is the human** as above. "
            "JARVIS in protocol headers is product branding, not a replacement for A). "
            "About yourself speak **I/me/my**, never **she/her** for the assistant."
        )
    else:
        tail = (
            "\n[FINAL — SAME NAME RULES]\n"
            "If any other text conflicts, use **A)** as the assistant name above. "
            "JARVIS in protocol headers is product branding, not a replacement for A). "
            "About yourself speak **I/me/my**, never **she/her** for the assistant."
        )
    return "\n" + body + tail


def format_memory_for_prompt(memory: dict | None) -> str:
    if not memory:
        return ""

    identity = memory.get("identity", {})
    if not isinstance(identity, dict):
        identity = {}

    assistant = _memory_entry_str(identity.get("assistant_name"))
    skip_identity = frozenset({"assistant_name"})
    id_fields = [
        "name",
        "age",
        "birthday",
        "city",
        "job",
        "language",
        "school",
        "nationality",
    ]

    human_nm = _memory_entry_str(identity.get("name"))

    chunks: list[str] = []
    if assistant:
        chunks.append(_registered_names_prompt(assistant=assistant, human_first=human_nm))
        chunks.append(
            "[ASSISTANT VOICE / PRODUCT]\n"
            f"The assistant introduces as «{assistant}». "
            "JARVIS in protocol titles is product branding, not a second assistant name.\n"
        )

    lines: list[str] = []
    skip_human_name_line = bool(assistant and human_nm)
    for field in id_fields:
        if field in skip_identity:
            continue
        entry = identity.get(field)
        if not entry:
            continue
        val = entry.get("value") if isinstance(entry, dict) else entry
        if not val:
            continue
        if field == "name":
            if skip_human_name_line:
                continue
            lines.append(
                f"Human user's name (for \"my name\" / self questions only): {val}"
            )
        else:
            lines.append(f"{field.title()}: {val}")
    for key, entry in identity.items():
        if key in id_fields or key in skip_identity:
            continue
        val = entry.get("value") if isinstance(entry, dict) else entry
        if val:
            lines.append(f"{key.replace('_', ' ').title()}: {val}")

    prefs = memory.get("preferences", {})
    if prefs:
        lines.append("")
        lines.append("Preferences:")
        for key, entry in list(prefs.items())[:15]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    projects = memory.get("projects", {})
    if projects:
        lines.append("")
        lines.append("Active Projects / Goals:")
        for key, entry in list(projects.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    rels = memory.get("relationships", {})
    if rels:
        lines.append("")
        lines.append("People in their life:")
        for key, entry in list(rels.items())[:10]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    wishes = memory.get("wishes", {})
    if wishes:
        lines.append("")
        lines.append("Wishes / Plans / Wants:")
        for key, entry in list(wishes.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    notes = memory.get("notes", {})
    if notes:
        lines.append("")
        lines.append("Other notes:")
        for key, entry in list(notes.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key}: {val}")

    if lines:
        chunks.append(
            "[WHAT YOU KNOW ABOUT THE HUMAN USER — use naturally, never recite like a list]\n"
            + "\n".join(lines)
        )

    if not chunks:
        return ""

    result = "\n\n".join(chunks)
    if len(result) > 2000:
        result = result[:1997] + "…"

    return result + "\n"

def remember(key: str, value: str, category: str = "notes") -> str:
    valid = {"identity", "preferences", "projects", "relationships", "wishes", "notes"}
    if category not in valid:
        category = "notes"
    update_memory({category: {key: {"value": value}}})
    return f"Remembered: {category}/{key} = {value}"


def forget(key: str, category: str = "notes") -> str:
    memory = load_memory()
    cat    = memory.get(category, {})
    if key in cat:
        del cat[key]
        memory[category] = cat
        save_memory(memory)
        return f"Forgotten: {category}/{key}"
    return f"Not found: {category}/{key}"


forget_memory = forget