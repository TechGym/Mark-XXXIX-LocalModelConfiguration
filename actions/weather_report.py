"""Live weather via Open-Meteo (no API key). Optional Google tab if configured."""

from __future__ import annotations

import json
import re
import time
import webbrowser
from collections.abc import Callable
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

_BAD_CITY_TOKENS = frozenset(
    {
        "your city",
        "your_city",
        "yourcity",
        "insert",
        "here",
        "unknown",
        "n/a",
        "na",
        "any city",
        "mycity",
        "my city",
        "the city",
        "local",
        "near me",
        "placeholder",
    }
)


def _scrub_city(raw: object) -> str:
    if not isinstance(raw, str):
        return ""
    t = raw.strip()
    if not t or len(t) > 120:
        return ""
    tl = t.lower().strip(" '\"")
    if tl in _BAD_CITY_TOKENS:
        return ""
    if "<" in t or ">" in t or "[" in t:
        return ""
    if "insert" in tl or "placeholder" in tl:
        return ""
    return t.strip()


def _wmo_summary(code: int) -> str:
    c = int(code)
    if c == 0:
        return "clear skies"
    if c == 1:
        return "mainly clear"
    if c == 2:
        return "partly cloudy"
    if c == 3:
        return "overcast"
    if c in (45, 48):
        return "fog"
    if c in (51, 53, 55, 56, 57):
        return "drizzle"
    if c in (61, 63, 65, 80, 81, 82):
        return "rain"
    if c in (66, 67):
        return "freezing rain"
    if c in (71, 73, 75, 77, 85, 86):
        return "snow"
    if c in (95, 96, 99):
        return "thunderstorms"
    if 1 <= c <= 99:
        return "variable conditions"
    return "current conditions"


def _fmt_num(v: object) -> str:
    try:
        x = float(v)
        if abs(x - round(x)) < 0.05:
            return str(int(round(x)))
        return str(round(x, 1))
    except (TypeError, ValueError):
        return str(v)


def _http_json(url: str, *, timeout: int = 18, retries: int = 3) -> Any:
    """GET JSON with short retries (Wi-Fi / corporate proxy flakiness)."""
    last: BaseException | None = None
    for attempt in range(max(1, retries)):
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": "Mark-XXXIX/1.0 (weather; https://open-meteo.com)",
                    "Accept": "application/json",
                },
            )
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except json.JSONDecodeError as e:
            last = e
            print(f"[Weather] JSON decode error for {url.split('?', 1)[0]}: {e}")
            break
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            last = e
            if attempt + 1 >= retries:
                break
            time.sleep(0.35 * (2**attempt))
    assert last is not None
    raise last


def _geocode_open_meteo(city: str) -> dict[str, Any] | None:
    q = urlencode({"name": city, "count": 1, "language": "en", "format": "json"})
    url = f"https://geocoding-api.open-meteo.com/v1/search?{q}"
    try:
        data = _http_json(url)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        print(f"[Weather] Open-Meteo geocode HTTP error for {city!r}: {e}")
        return None
    results = data.get("results")
    if not isinstance(results, list) or not results:
        return None
    r0 = results[0]
    if not isinstance(r0, dict):
        return None
    lat, lon = r0.get("latitude"), r0.get("longitude")
    if lat is None or lon is None:
        return None
    name = str(r0.get("name") or city).strip() or city
    parts: list[str] = [name]
    a1 = r0.get("admin1")
    if isinstance(a1, str) and a1.strip():
        parts.append(a1.strip())
    cc = r0.get("country_code")
    if isinstance(cc, str) and cc.strip():
        parts.append(cc.strip())
    label = ", ".join(parts)
    return {"lat": float(lat), "lon": float(lon), "label": label}


def _geocode_nominatim(city: str) -> dict[str, Any] | None:
    """
    OSM Nominatim fallback when Open-Meteo geocoding is empty or unreachable.

    See https://operations.osmfoundation.org/policies/nominatim/ — one request
    per second, identifiable User-Agent.
    """
    query = (city or "").strip()
    if not query:
        return None
    time.sleep(1.0)
    params = urlencode({"q": query, "format": "json", "limit": "1"})
    url = f"https://nominatim.openstreetmap.org/search?{params}"
    try:
        req = Request(
            url,
            headers={
                "User-Agent": "Mark-XXXIX/1.0 (weather geocode fallback; no bulk use)",
                "Accept": "application/json",
            },
        )
        with urlopen(req, timeout=22) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        print(f"[Weather] Nominatim geocode failed for {query!r}: {e}")
        return None
    if not isinstance(data, list) or not data:
        return None
    row = data[0]
    if not isinstance(row, dict):
        return None
    try:
        lat = float(row["lat"])
        lon = float(row["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    nm = str(row.get("name") or "").strip()
    disp = str(row.get("display_name") or "").strip()
    if nm:
        label = nm
    elif disp:
        label = ", ".join(p.strip() for p in disp.split(",")[:3] if p.strip())
    else:
        label = query
    return {"lat": lat, "lon": lon, "label": label}


def _normalize_spoken_place_typos(city: str) -> str:
    """Fix common STT comma splices before geocoding (e.g. ``Lehigh, Acres`` → Lehigh Acres)."""
    c = (city or "").strip()
    if not c:
        return c
    # Whisper often inserts a comma between tokens of the CDP name "Lehigh Acres".
    c = re.sub(r"(?i)\blehigh\s*,\s*acres\b", "Lehigh Acres", c)
    # "Lea Acres" — dropped syllable in "Lehigh" (common STT error for this CDP).
    c = re.sub(r"(?i)\blea\s+acres\b", "Lehigh Acres", c)
    return c


def _geocode_variants(city: str) -> list[str]:
    """Try the query as-is, then common Open-Meteo spelling alternates (deduped)."""
    c = _normalize_spoken_place_typos((city or "").strip())
    if not c:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        t = s.strip()
        if not t or t.lower() in seen:
            return
        seen.add(t.lower())
        out.append(t)

    _add(c)
    # CDP is usually "Lehigh Acres" in gazetteers; "Lehigh, FL" often misses.
    if re.search(r"\blehigh\b", c, re.IGNORECASE) and "acres" not in c.lower():
        _add(re.sub(r"\blehigh\b", "Lehigh Acres", c, flags=re.IGNORECASE))
    return out


def _geocode_open_meteo_best(city: str) -> dict[str, Any] | None:
    base = (city or "").strip()
    if not base:
        return None
    for variant in _geocode_variants(base):
        geo = _geocode_open_meteo(variant)
        if geo:
            if variant.strip().lower() != base.lower():
                print(f"[Weather] geocode: used {variant!r} (retry) for query {base!r}")
            return geo
    geo = _geocode_nominatim(base)
    if geo:
        print(f"[Weather] geocode: Nominatim fallback for {base!r} -> {geo.get('label')!r}")
        return geo
    print(f"[Weather] geocode: no results (Open-Meteo + Nominatim) for {base!r}")
    return None


def _forecast_open_meteo(lat: float, lon: float, *, imperial: bool) -> dict[str, Any]:
    params: dict[str, str | float] = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(
            [
                "temperature_2m",
                "apparent_temperature",
                "weather_code",
                "wind_speed_10m",
                "relative_humidity_2m",
            ]
        ),
    }
    if imperial:
        params["temperature_unit"] = "fahrenheit"
        params["wind_speed_unit"] = "mph"
    else:
        params["temperature_unit"] = "celsius"
        params["wind_speed_unit"] = "kmh"
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    try:
        return _http_json(url)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        print(f"[Weather] Open-Meteo forecast HTTP error at {lat},{lon}: {e}")
        return {}


def _format_report(label: str, cur: dict[str, Any], *, imperial: bool) -> str:
    t = cur.get("temperature_2m")
    feel = cur.get("apparent_temperature")
    code = int(cur.get("weather_code") or 0)
    wind = cur.get("wind_speed_10m")
    rh = cur.get("relative_humidity_2m")
    unit = "°F" if imperial else "°C"
    wunit = "mph" if imperial else "km/h"
    sky = _wmo_summary(code)
    t_s = _fmt_num(t)
    bits = [f"in {label} it's {t_s}{unit} and {sky}"]
    if feel is not None and t is not None:
        try:
            if abs(float(feel) - float(t)) > 0.4:
                bits.append(f"feels like {_fmt_num(feel)}{unit}")
        except (TypeError, ValueError):
            pass
    if wind is not None:
        bits.append(f"wind {_fmt_num(wind)} {wunit}")
    if rh is not None:
        try:
            bits.append(f"humidity {int(round(float(rh)))}%")
        except (TypeError, ValueError):
            pass
    return ", ".join(bits) + "."


def _forecast_body_for_city(
    city_query: str,
    *,
    imperial: bool,
) -> tuple[str, str] | None:
    """Return ``(sentence, resolved_label)`` or ``None`` if geocode/forecast fails."""
    q = city_query.strip()
    if not q:
        return None
    geo = _geocode_open_meteo_best(q)
    if not geo:
        return None
    fc = _forecast_open_meteo(float(geo["lat"]), float(geo["lon"]), imperial=imperial)
    current = fc.get("current") or {}
    if not isinstance(current, dict) or current.get("temperature_2m") is None:
        return None
    label = str(geo["label"])
    body = _format_report(label, current, imperial=imperial)
    return (body, label)


def weather_action(
    parameters: dict,
    player=None,
    session_memory=None,
    speak: Optional[Callable[[str], None]] = None,
) -> str:
    from mark_llm_settings import (
        get_weather_default_cities,
        get_weather_open_browser,
        get_weather_use_imperial_units,
    )

    # Ollama may send ``{"city": null}`` — ``.get("city", "")`` would return None.
    city_raw = parameters.get("city") or ""
    when = str(parameters.get("time") or "today").strip() or "today"

    city = _scrub_city(city_raw)
    if city:
        targets = [city]
    else:
        targets = get_weather_default_cities()
    if not targets:
        msg = (
            "Sir, I need a town to look up. Name a city in your question, or set "
            "``weather_cities`` (array) or ``weather_city`` in config/api_keys.json."
        )
        _log(msg, player)
        return msg

    imperial = get_weather_use_imperial_units()

    try:
        bodies: list[str] = []
        labels: list[str] = []
        for t in targets:
            got = _forecast_body_for_city(t, imperial=imperial)
            if got:
                bodies.append(got[0])
                labels.append(got[1])
        if not bodies:
            if len(targets) == 1:
                msg = (
                    f'Sir, I could not find or read weather for "{targets[0]}". '
                    "Try a nearby city name."
                )
            else:
                tried = "; ".join(targets)
                msg = (
                    f"Sir, I could not fetch weather for the configured places ({tried}). "
                    "Open-Meteo geocoding and the OSM Nominatim fallback both failed; check "
                    "spelling in ``weather_cities``, try larger nearby towns, or confirm "
                    "this PC can reach https://geocoding-api.open-meteo.com and "
                    "https://nominatim.openstreetmap.org (firewall / VPN). "
                    "See terminal [Weather] lines."
                )
            _log(msg, player)
            return msg
        if len(bodies) == 1:
            msg = f"Sir, {bodies[0]}"
        else:
            msg = "Sir, " + " Also, ".join(bodies)
        label = labels[0] if len(labels) == 1 else " — ".join(labels)
    except (HTTPError, URLError, OSError, TimeoutError, ValueError, TypeError) as e:
        msg = f"Sir, the weather service failed: {e}"
        _log(msg, player)
        return msg

    _log(msg, player)
    if speak:
        try:
            speak(msg if len(msg) <= 900 else msg[:897] + "…")
        except Exception:
            pass

    if get_weather_open_browser():
        try:
            search_query = f"weather {when} " + " ".join(labels)
            url = f"https://www.google.com/search?q={quote_plus(search_query)}"
            webbrowser.open(url)
        except Exception:
            pass

    if session_memory:
        try:
            session_memory.set_last_search(
                query="weather:" + ";".join(targets), response=msg[:500]
            )
        except Exception:
            pass

    return msg


def _log(message: str, player=None) -> None:
    print(f"[Weather] {message}")
    if player:
        try:
            player.write_log(f"JARVIS: {message}")
        except Exception:
            pass
