import re
import urllib.parse
from typing import List, Dict, Tuple, Any

import httpx
from bs4 import BeautifulSoup
import subprocess


# Simple in-memory caches to avoid repeating network calls
_SEARCH_CACHE: Dict[Tuple[str, int], Dict] = {}
_URL_CACHE: Dict[Tuple[str, int], Dict] = {}

async def _ddg_search_html(q: str, k: int = 5) -> List[Dict[str, str]] | Dict[str, str]:
    """Scrape DuckDuckGo's HTML results and return [{title,url,snippet}, ...]."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            r = await client.get(
                "https://duckduckgo.com/html/",
                params={"q": q},
                headers=headers,
            )
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
    except httpx.HTTPError as e:
        return {"error": f"web search failed: {e}"}
    except Exception as e:
        return {"error": f"web search failed: {e}"}
    items = []
    for res in soup.select("div.result")[:k]:
        a = res.select_one("a.result__a")
        if not a:
            continue
        href = a.get("href", "")
        if "uddg=" in href:
            try:
                qs = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query)
                href = urllib.parse.unquote(qs.get("uddg", [href])[0])
            except Exception:
                pass
        snippet_el = res.select_one(".result__snippet")
        items.append(
            {
                "title": a.get_text(" ", strip=True),
                "url": href,
                "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
            }
        )
    return items

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

async def _open_and_extract(url: str, max_chars: int = 6000) -> Dict[str, str]:
    """Fetch a URL and return {'title','url','text'} with trimmed, readable text."""
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    main = soup.find(["article", "main"]) or soup.body or soup
    parts = []
    for el in main.find_all(["h1", "h2", "h3", "p", "li"], limit=1200):
        txt = el.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
    text = _clean_text(" ".join(parts))
    if len(text) > max_chars:
        text = text[:max_chars] + " …"
    return {"title": title, "url": url, "text": text}

async def web_search(query: str, k: int = 5) -> Dict:
    """Search the web for recent or factual info and return top results."""
    key = (query, k)
    if key not in _SEARCH_CACHE:
        try:
            results = await _ddg_search_html(query, k)
            if isinstance(results, dict) and results.get("error"):
                return results
            _SEARCH_CACHE[key] = {"results": results, "source": "duckduckgo_html"}
        except httpx.HTTPError as e:
            return {"error": f"web search failed: {e}"}
        except Exception as e:
            return {"error": f"web search failed: {e}"}
    return _SEARCH_CACHE[key]

async def open_url(url: str, max_chars: int = 6000) -> Dict:
    """Open a URL and return a concise text extract for summarization."""
    key = (url, max_chars)
    if key not in _URL_CACHE:
        page = await _open_and_extract(url, max_chars=max_chars)
        _URL_CACHE[key] = {"page": page}
    return _URL_CACHE[key]


# ----------------- Local utility tools -----------------

async def eval_expr(expr: str) -> Dict[str, Any]:
    """Evaluate a Python expression and return the result."""
    try:
        # Evaluate with no builtins for a bit of safety
        result = eval(expr, {"__builtins__": {}})
        return {"result": repr(result)}
    except Exception as e:
        return {"error": str(e)}


async def execute(code: str) -> Dict[str, Any]:
    """Execute a Python code snippet and capture stdout/stderr."""
    try:
        proc = subprocess.run(
            ["python", "-"],
            input=code,
            text=True,
            capture_output=True,
            timeout=10,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    except Exception as e:
        return {"error": str(e)}


async def read_file(path: str) -> Dict[str, Any]:
    """Read a text file and return its contents."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except Exception as e:
        return {"error": str(e)}


async def write_file(path: str, contents: str) -> Dict[str, Any]:
    """Write text to a file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(contents)
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


# Terminal session state (single session for simplicity)
_TERMINAL_OPEN = False


async def terminal_open() -> Dict[str, Any]:
    """Open a pseudo terminal session."""
    global _TERMINAL_OPEN
    _TERMINAL_OPEN = True
    return {"ok": True}


async def terminal_run(cmd: str) -> Dict[str, Any]:
    """Run a shell command in the terminal session."""
    if not _TERMINAL_OPEN:
        return {"error": "terminal not open"}
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    except Exception as e:
        return {"error": str(e)}


async def terminal_terminate() -> Dict[str, Any]:
    """Terminate the pseudo terminal session."""
    global _TERMINAL_OPEN
    _TERMINAL_OPEN = False
    return {"ok": True}


# Notes and user preferences storage
_NOTES: Dict[str, str] = {}
_USER_PREFS: Dict[str, str] = {}


async def notes_write(key: str, content: str) -> Dict[str, Any]:
    _NOTES[key] = content
    return {"ok": True}


async def notes_list() -> Dict[str, Any]:
    return {"keys": list(_NOTES.keys())}


async def notes_read(key: str) -> Dict[str, Any]:
    if key in _NOTES:
        return {"content": _NOTES[key]}
    return {"error": "not found"}


async def user_prefs_write(key: str, content: str) -> Dict[str, Any]:
    _USER_PREFS[key] = content
    return {"ok": True}


async def user_prefs_list() -> Dict[str, Any]:
    return {"keys": list(_USER_PREFS.keys())}


async def user_prefs_read(key: str) -> Dict[str, Any]:
    if key in _USER_PREFS:
        return {"content": _USER_PREFS[key]}
    return {"error": "not found"}

