import re
import urllib.parse
from typing import List, Dict, Tuple

import httpx
from bs4 import BeautifulSoup

# Simple in-memory caches to avoid repeating network calls
_SEARCH_CACHE: Dict[Tuple[str, int], Dict] = {}
_URL_CACHE: Dict[Tuple[str, int], Dict] = {}

async def _ddg_search_html(q: str, k: int = 5) -> List[Dict[str, str]]:
    """Scrape DuckDuckGo's HTML results and return [{title,url,snippet}, ...]."""
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(
            "https://duckduckgo.com/html/",
            params={"q": q},
            headers=headers,
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
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
        results = await _ddg_search_html(query, k)
        _SEARCH_CACHE[key] = {"results": results, "source": "duckduckgo_html"}
    return _SEARCH_CACHE[key]

async def open_url(url: str, max_chars: int = 6000) -> Dict:
    """Open a URL and return a concise text extract for summarization."""
    key = (url, max_chars)
    if key not in _URL_CACHE:
        page = await _open_and_extract(url, max_chars=max_chars)
        _URL_CACHE[key] = {"page": page}
    return _URL_CACHE[key]
