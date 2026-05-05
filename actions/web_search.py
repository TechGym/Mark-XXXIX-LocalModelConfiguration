# web_search.py
import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR        = _get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"


def _get_api_key() -> str:
    with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _gemini_search(query: str) -> str:
    from google import genai

    client   = genai.Client(api_key=_get_api_key())
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config={"tools": [{"google_search": {}}]},
    )

    text = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            text += part.text

    text = text.strip()
    if not text:
        raise ValueError("Gemini returned an empty response.")
    return text


def _strip_html_snippet(s: str, *, limit: int = 400) -> str:
    if not s:
        return ""
    t = re.sub(r"<[^>]+>", " ", s)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit]


def _query_suggests_news_rss_fallback(query: str) -> bool:
    """When DDG returns nothing, RSS feeds often still work (different host / rate limits)."""
    ql = (query or "").lower()
    needles = (
        "world",
        "headline",
        "news",
        "global",
        "international",
        "breaking",
        "today",
        "current events",
    )
    return any(n in ql for n in needles)


def _rss_channel_items(
    feed_url: str, *, max_items: int = 10, timeout: int = 20
) -> list[dict]:
    import xml.etree.ElementTree as ET

    req = Request(
        feed_url,
        headers={
            "User-Agent": (
                "Mark-XXXIX/1.0 (news RSS fallback; "
                "+https://github.com/denalidao/Mark-XXXIX)"
            ),
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (HTTPError, URLError, OSError, TimeoutError) as e:
        print(f"[WebSearch] RSS fetch failed {feed_url!r}: {e}")
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        print(f"[WebSearch] RSS parse error for {feed_url!r}: {e}")
        return []
    channel = root.find("channel")
    if channel is None:
        return []
    out: list[dict] = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "").strip()
        if not title:
            continue
        out.append(
            {
                "title": title,
                "snippet": _strip_html_snippet(desc),
                "url": link or feed_url,
            }
        )
        if len(out) >= max_items:
            break
    return out


def _news_rss_fallback_results(query: str, max_results: int = 8) -> list[dict]:
    if not _query_suggests_news_rss_fallback(query):
        return []
    for feed_url in (
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.rt.com/rss/",
    ):
        items = _rss_channel_items(feed_url, max_items=max_results)
        if items:
            print(f"[WebSearch] RSS headlines from {feed_url}")
            return items
    return []


# Hostname in a user/model query: try ``site:``, bare host, then direct HTTP excerpt.
# One or more ``label.`` segments, then a known TLD (``denalidao.com`` matches).
_DOMAIN_IN_QUERY = re.compile(
    r"(?i)\b((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:com|org|net|io|ai|dev|gov|co|app|xyz|tech|news|blog|finance|us|uk))\b"
)


def extract_primary_domain_from_query(query: str) -> str | None:
    m = _DOMAIN_IN_QUERY.search(query or "")
    return m.group(1).lower() if m else None


def _collect_ddg_with_domain_variants(query: str, *, max_results: int = 8) -> list[dict]:
    """Merge DDG rows for the raw query plus ``site:host`` and bare ``host`` when a domain appears."""
    seen: set[str] = set()
    out: list[dict] = []

    def _add(rows: list[dict]) -> None:
        for row in rows:
            key = (row.get("url") or "").strip() or (row.get("title") or "").strip()[:160]
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(row)
            if len(out) >= max_results:
                return

    _add(_ddg_search(query, max_results=max_results))
    host = extract_primary_domain_from_query(query)
    if host:
        _add(_ddg_search(f"site:{host}", max_results=max_results))
        if len(out) < max_results:
            _add(_ddg_search(host, max_results=max_results))
    return out[:max_results]


def _is_safe_http_url(url: str) -> bool:
    u = (url or "").strip()
    if not u or len(u) > 2048:
        return False
    low = u.lower()
    return low.startswith("https://") or low.startswith("http://")


def _html_to_plain_text(html: str, *, max_chars: int) -> str:
    """Cheap HTML → text (no extra deps)."""
    if not html:
        return ""
    t = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    t = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", t)
    t = re.sub(r"(?is)<br\s*/?>", "\n", t)
    t = re.sub(r"(?is)</p>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t).strip()
    return t[:max_chars]


def fetch_url_as_text(url: str, *, timeout: int = 25, max_bytes: int = 600_000, max_chars: int = 14_000) -> str:
    """
    Fetch a single public web page and return plain-text excerpt (for ``mode: fetch``).
    """
    if not _is_safe_http_url(url):
        return "Invalid or unsupported URL — only http(s) links up to 2048 chars are allowed."
    try:
        req = Request(
            url.strip(),
            headers={
                "User-Agent": (
                    "Mark-XXXIX/1.0 (read page for assistant; "
                    "+https://github.com/denalidao/Mark-XXXIX)"
                ),
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            },
        )
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
        html = raw.decode("utf-8", errors="replace")
    except (HTTPError, URLError, OSError, TimeoutError, ValueError) as e:
        return f"Could not fetch URL: {e}"

    title = ""
    t_m = re.search(r"(?is)<title[^>]*>([^<]{1,400})", html)
    if t_m:
        title = _strip_html_snippet(t_m.group(1))[:300]
    body = _html_to_plain_text(html, max_chars=max_chars)
    if not body and not title:
        return "Fetched the page but could not extract readable text."
    parts = [f"URL: {url.strip()}"]
    if title:
        parts.append(f"Title: {title}")
    parts.append("")
    parts.append(body or "(empty body after stripping HTML)")
    return "\n".join(parts).strip()


def _fetch_homepage_snippet(hostname: str, *, timeout: int = 18, max_bytes: int = 400_000) -> list[dict]:
    """
    Last resort: GET ``https://{host}/`` (and ``www``) and build one pseudo search row
    so the model can summarize a site DDG often misses.
    """
    host = (hostname or "").strip().lower().rstrip(".")
    if not host or "." not in host:
        return []
    for url in (f"https://{host}/", f"https://www.{host}/"):
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": (
                        "Mark-XXXIX/1.0 (homepage excerpt for assistant; "
                        "+https://github.com/denalidao/Mark-XXXIX)"
                    ),
                    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                },
            )
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raw = raw[:max_bytes]
            html = raw.decode("utf-8", errors="replace")
        except (HTTPError, URLError, OSError, TimeoutError, ValueError) as e:
            print(f"[WebSearch] HTTP homepage fetch failed {url!r}: {e}")
            continue
        title = ""
        t_m = re.search(r"(?is)<title[^>]*>([^<]{1,280})", html)
        if t_m:
            title = _strip_html_snippet(t_m.group(1))[:240]
        snippet = ""
        d_m = re.search(
            r'(?is)<meta\s+[^>]*name\s*=\s*["\']description["\'][^>]*\s+content\s*=\s*["\']([^"\'<]{1,500})',
            html,
        ) or re.search(
            r'(?is)<meta\s+[^>]*content\s*=\s*["\']([^"\'<]{1,500})[^>]*name\s*=\s*["\']description["\']',
            html,
        )
        if d_m:
            snippet = _strip_html_snippet(d_m.group(1))
        if len(snippet) < 40:
            p_m = re.search(r"(?is)<p[^>]*>([^<]{25,800})", html)
            if p_m:
                snippet = _strip_html_snippet(p_m.group(1))
        if not title:
            title = host
        if not snippet:
            snippet = "(No plain-text excerpt could be pulled from the homepage HTML.)"
        return [
            {
                "title": f"{title} (homepage)",
                "snippet": snippet[:900],
                "url": url,
            }
        ]
    return []


def _ddg_search(query: str, max_results: int = 6) -> list[dict]:
    """Fetch web snippets via DuckDuckGo.

    The library default ``backend="auto"`` often returns nothing while
    ``html`` / ``lite`` still work, so we try several backends then news.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    seen: set[str] = set()
    out: list[dict] = []

    def _take_row(r: dict) -> None:
        url = (r.get("href") or r.get("url") or "").strip()
        key = url or ((r.get("title") or "").strip()[:120])
        if not key or key in seen:
            return
        seen.add(key)
        out.append(
            {
                "title": r.get("title", ""),
                "snippet": r.get("body", r.get("snippet", "")),
                "url": url or r.get("href", ""),
            }
        )

    with DDGS() as ddgs:
        for backend in ("html", "lite", "auto"):
            try:
                batch = list(
                    ddgs.text(
                        query,
                        max_results=max_results,
                        backend=backend,
                        safesearch="off",
                    )
                )
            except Exception as exc:
                print(f"[WebSearch] ⚠️ DDG text backend={backend!r}: {exc}")
                batch = []
            for r in batch:
                _take_row(r)
                if len(out) >= max_results:
                    return out[:max_results]

        if len(out) < max_results:
            try:
                for r in ddgs.news(
                    query,
                    max_results=max_results,
                    safesearch="off",
                    region="us-en",
                ):
                    _take_row(
                        {
                            "title": r.get("title", ""),
                            "body": r.get("body", ""),
                            "href": r.get("url", ""),
                        }
                    )
                    if len(out) >= max_results:
                        break
            except Exception as exc:
                print(f"[WebSearch] ⚠️ DDG news fallback: {exc}")

    return out[:max_results]


def _ddg_news_only(query: str, *, max_results: int = 10) -> list[dict]:
    """Headlines only (DDG news index), for ``mode: news``."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    out: list[dict] = []
    seen: set[str] = set()
    with DDGS() as ddgs:
        try:
            for r in ddgs.news(
                query,
                max_results=max_results,
                safesearch="off",
                region="us-en",
            ):
                url = (r.get("url") or "").strip()
                key = url or (r.get("title") or "").strip()[:120]
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": url,
                    }
                )
                if len(out) >= max_results:
                    break
        except Exception as exc:
            print(f"[WebSearch] ⚠️ DDG news-only: {exc}")
    return out


def _format_ddg(query: str, results: list[dict]) -> str:
    if not results:
        return (
            f"No search results were returned for: {query}\n"
            "DuckDuckGo may be rate-limiting or blocking automated requests; "
            "wait a minute and retry, or check network and VPN, sir.\n\n"
            "For the assistant: tell the user briefly that no headline snippets came back "
            "this round (often rate limits or the query phrasing). Suggest they try again "
            "in a minute. Do not invent articles or sources; do not claim their home internet "
            "is broken unless they said so."
        )

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        if r.get("title"):   lines.append(f"{i}. {r['title']}")
        if r.get("snippet"): lines.append(f"   {r['snippet']}")
        if r.get("url"):     lines.append(f"   {r['url']}")
        lines.append("")
    return "\n".join(lines).strip()

def _compare(items: list[str], aspect: str) -> str:
    from mark_llm_settings import is_ollama_mode

    query = (
        f"Compare {', '.join(items)} in terms of {aspect}. "
        "Give specific facts and data."
    )
    if is_ollama_mode():
        print("[WebSearch] 🦙 Ollama mode — compare via DuckDuckGo only.")
    else:
        try:
            return _gemini_search(query)
        except Exception as e:
            print(f"[WebSearch] ⚠️ Gemini compare failed: {e} — falling back to DDG")

    # DDG fallback: fetch results per item and merge
    all_results: dict[str, list] = {}
    for item in items:
        try:
            all_results[item] = _ddg_search(f"{item} {aspect}", max_results=3)
        except Exception:
            all_results[item] = []

    lines = [f"Comparison — {aspect.upper()}", "─" * 40]
    for item in items:
        lines.append(f"\n▸ {item}")
        for r in all_results.get(item, [])[:2]:
            if r.get("snippet"):
                lines.append(f"  • {r['snippet']}")
    return "\n".join(lines)

def web_search(
    parameters:     dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    query = (params.get("query") or "").strip()
    mode = (params.get("mode") or "search").lower().strip()
    items = params.get("items", [])
    aspect = (params.get("aspect") or "general").strip() or "general"
    url = (params.get("url") or "").strip()
    max_chars = params.get("max_chars")
    try:
        max_chars_i = int(max_chars) if max_chars is not None else 14_000
    except (TypeError, ValueError):
        max_chars_i = 14_000
    max_chars_i = max(2_000, min(max_chars_i, 40_000))

    if mode == "fetch":
        if not url:
            return "For mode **fetch**, provide a non-empty **url** (https://…)."
        if player:
            player.write_log(f"[Search] fetch {url[:120]}")
        print(f"[WebSearch] 📄 Fetch URL mode: {url!r}")
        return fetch_url_as_text(url, max_chars=max_chars_i)

    if mode == "news":
        if not query:
            return "For mode **news**, provide a **query** (topic or outlet)."
        if player:
            player.write_log(f"[Search] news {query}")
        print(f"[WebSearch] 📰 News mode: {query!r}")
        rows = _ddg_news_only(query, max_results=10)
        if not rows and _query_suggests_news_rss_fallback(query):
            rows = _news_rss_fallback_results(query, max_results=10)
        return _format_ddg(query, rows)

    if not query and not items:
        return "Please provide a search **query** (or use mode **fetch** with **url**)."

    if items and mode != "compare":
        mode = "compare"

    if player:
        player.write_log(f"[Search] {query or ', '.join(items)}")

    print(f"[WebSearch] 🔍 Query: {query!r}  Mode: {mode}")

    try:
        if mode == "compare" and items:
            print(f"[WebSearch] 📊 Comparing: {items}")
            result = _compare(items, aspect)
            print("[WebSearch] ✅ Compare done.")
            return result

        from mark_llm_settings import is_ollama_mode

        if is_ollama_mode():
            print(
                "[WebSearch] 🦙 Ollama chat mode — using DuckDuckGo (local model has no web; "
                "this fetches live snippets for you to summarize)."
            )
            results = _collect_ddg_with_domain_variants(query, max_results=8)
            if not results and _query_suggests_news_rss_fallback(query):
                rss = _news_rss_fallback_results(query)
                if rss:
                    results = rss
            host = extract_primary_domain_from_query(query)
            if not results and host:
                page_rows = _fetch_homepage_snippet(host)
                if page_rows:
                    print(f"[WebSearch] HTTP homepage excerpt for {host}")
                    results = page_rows
            result = _format_ddg(query, results)
            print(f"[WebSearch] ✅ {len(results)} snippet(s) for Ollama.")
            return result

        print("[WebSearch] 🌐 Trying Gemini (Google Search)…")
        try:
            result = _gemini_search(query)
            print("[WebSearch] ✅ Gemini OK.")
            return result
        except Exception as e:
            print(f"[WebSearch] ⚠️ Gemini failed ({e}) — trying DDG...")
            results = _collect_ddg_with_domain_variants(query, max_results=8)
            host = extract_primary_domain_from_query(query)
            if not results and host:
                page_rows = _fetch_homepage_snippet(host)
                if page_rows:
                    print(f"[WebSearch] HTTP homepage excerpt for {host}")
                    results = page_rows
            result = _format_ddg(query, results)
            print(f"[WebSearch] ✅ DDG (+domain variants): {len(results)} result(s).")
            return result

    except Exception as e:
        print(f"[WebSearch] ❌ All backends failed: {e}")
        return f"Search failed, sir: {e}"