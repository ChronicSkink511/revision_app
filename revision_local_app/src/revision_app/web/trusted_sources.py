from __future__ import annotations

import json
import re
from html import unescape
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen


DEFAULT_TRUSTED_DOMAINS = {
    "wikipedia.org",
    "nist.gov",
    "nasa.gov",
    "asme.org",
    "ieee.org",
    "mit.edu",
}



def _normalize_domain(value: str) -> str:
    return value.strip().lower().lstrip(".")



def _is_trusted(url: str, allowed_domains: set[str]) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False

    if not host:
        return False

    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)



def _clean_html_text(html: str, max_chars: int = 1800) -> str:
    text = re.sub(r"<script[\\s\\S]*?</script>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\\s\\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text[:max_chars]



def _fetch_url_text(url: str, timeout: int = 8, max_chars: int = 1800) -> str:
    request = Request(url, headers={"User-Agent": "RevisionApp/1.0"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(400000)
    html = raw.decode("utf-8", errors="ignore")
    return _clean_html_text(html, max_chars=max_chars)



def _search_wikipedia(query: str, limit: int = 2) -> list[str]:
    endpoint = (
        "https://en.wikipedia.org/w/api.php?action=opensearch&search="
        f"{quote_plus(query)}&limit={limit}&namespace=0&format=json"
    )
    request = Request(endpoint, headers={"User-Agent": "RevisionApp/1.0"})
    with urlopen(request, timeout=8) as response:
        payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    titles = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    urls = payload[3] if isinstance(payload, list) and len(payload) > 3 else []
    if isinstance(urls, list) and urls:
        return [u for u in urls if isinstance(u, str)]
    return [f"https://en.wikipedia.org/wiki/{quote_plus(t.replace(' ', '_'))}" for t in titles]



def gather_trusted_web_context(
    question: str,
    user_urls: list[str] | None = None,
    extra_domains: list[str] | None = None,
    max_sources: int = 3,
) -> list[dict]:
    allowed_domains = set(DEFAULT_TRUSTED_DOMAINS)
    if extra_domains:
        allowed_domains.update(_normalize_domain(d) for d in extra_domains if d.strip())

    candidates: list[str] = []
    if user_urls:
        candidates.extend([u.strip() for u in user_urls if u.strip()])

    # Add lightweight trusted discovery from Wikipedia when no URLs were supplied.
    if not candidates:
        try:
            candidates.extend(_search_wikipedia(question, limit=2))
        except Exception:
            pass

    results: list[dict] = []
    seen: set[str] = set()
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        if not _is_trusted(url, allowed_domains):
            continue

        try:
            text = _fetch_url_text(url)
            if not text:
                continue
            results.append({"source": url, "text": text})
        except Exception:
            continue

        if len(results) >= max_sources:
            break

    return results
