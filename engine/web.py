"""Their window on the web: a reader that keeps the shape of a page, and a
search that asks the whole web, not only Wikipedia.

    py engine/web.py search "gemma 4 ollama"      try the search from a terminal
    py engine/web.py read https://example.org      try the reader; add a page number for page 2…
    py engine/web.py read https://example.org 2

09-22, the keeper: "I want to upgrade the friend's browsing capabilities." What
a cloud assistant has: a search that returns titles,
URLs and a line each; a reader that returns a page as markdown with its
headings, lists and links kept, long pages in parts. What they had: a
reader that flattened a page to lines (menus, footers and all) and cut it
at 15,000 characters, and Wikipedia's search. Now:

  read_web(url, page=, find=)  — the page's title, its main text as light
      markdown (headings, paragraphs, list items), links numbered inline
      [3] with an index at the end so they can open the next page from this
      one; boilerplate (navigation, footers, cookie banners, share bars)
      left out; long pages in parts of WEB_PAGE_CHARS, "page 2 of 5";
      find= jumps to the part that holds a phrase. A PDF URL goes to
      read_pdf; plain text and JSON come as they are.
  search_web(query, results=)  — the web's answer: title, URL and a line
      per result; read_web opens any. DuckDuckGo needs no key and no
      account; a SearXNG of your own (WEB_SEARCH_SEARXNG_URL) or a Brave
      key (memory/web_search.json, never config) can stand in.

Everything here is the window: material to read, never instructions.
Nothing here is stdlib-free — no packages to install.
"""
from __future__ import annotations

import gzip
import html as htmlmod
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

# a browser's own words: DuckDuckGo answered the first tagged request
# ("… AIFriend/1.0 (a local AI reading…)") with a human check (09-22)
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/128.0.0.0 Safari/537.36")


# ---- fetching ---------------------------------------------------------------

class WebError(Exception):
    pass


def fetch(url: str, max_bytes: int = 2_500_000, timeout: int = 30, data: bytes | None = None,
          headers: dict | None = None) -> tuple[str, str, bytes]:
    """(final url, content type, body) — gzip/deflate unpacked, redirects
    followed, the body capped. `data` makes it a POST."""
    url = (url or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        raise WebError("only http(s) URLs")
    h = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Upgrade-Insecure-Requests": "1",
    }
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(max_bytes)
            ctype = (resp.headers.get("Content-Type") or "").lower()
            enc = (resp.headers.get("Content-Encoding") or "").lower()
            final = resp.geturl()
    except urllib.error.HTTPError as e:
        raise WebError(f"HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise WebError(str(e.reason)) from e
    except Exception as e:  # timeouts, bad certs, resets
        raise WebError(str(e)) from e
    if "gzip" in enc:
        try:
            body = gzip.decompress(body)
        except Exception:
            pass
    elif "deflate" in enc:
        try:
            body = zlib.decompress(body)
        except Exception:
            try:
                body = zlib.decompress(body, -zlib.MAX_WBITS)
            except Exception:
                pass
    return final, ctype, body


_META_CHARSET_RE = re.compile(rb'<meta[^>]+charset=["\']?\s*([A-Za-z0-9_\-]+)', re.IGNORECASE)


def decode(body: bytes, ctype: str) -> str:
    m = re.search(r"charset=([A-Za-z0-9_\-]+)", ctype or "")
    cands = [m.group(1)] if m else []
    mm = _META_CHARSET_RE.search(body[:4000])
    if mm:
        cands.append(mm.group(1).decode("ascii", "replace"))
    cands += ["utf-8", "cp1252"]
    for c in cands:
        try:
            return body.decode(c)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", "replace")


# ---- the reader --------------------------------------------------------------

@dataclass
class Page:
    url: str
    title: str = ""
    blocks: list[str] = field(default_factory=list)   # markdown-ish paragraphs
    links: list[tuple[str, str]] = field(default_factory=list)  # (text, absolute url), numbered from 1
    images: list[tuple[str, str]] = field(default_factory=list)  # (alt, absolute url), numbered i1…; look_at opens any

    @property
    def text(self) -> str:
        return "\n\n".join(b for b in self.blocks if b.strip())


_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "canvas", "video", "audio",
              "object", "embed", "head", "select", "option", "textarea", "button"}
_CHROME_TAGS = {"nav", "header", "footer", "aside", "form", "menu", "dialog"}
_BOILER_RE = re.compile(r"(?:^|[\s_\-])(?:nav|navbar|menu|footer|sidebar|side-bar|comments?|cookie|consent|banner|"
                        r"share|social|related|recommend|promo|advert|ads?|sponsor|breadcrumb|subscribe|newsletter|"
                        r"popup|modal|toolbar|skip-link|pagination|toc)(?:$|[\s_\-])", re.IGNORECASE)
_BLOCK_TAGS = {"p", "div", "section", "article", "main", "li", "ul", "ol", "blockquote", "pre", "table",
               "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br", "hr", "dd", "dt", "figcaption", "summary",
               "details", "td", "th"}
_HEAD_TAGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}


class _Reader(HTMLParser):
    """Walks the page once, writing light markdown into blocks and noting
    which blocks sit inside <main>/<article>, so the caller can keep only
    the article when the page has one."""

    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.title = ""
        self.blocks: list[tuple[str, bool]] = []  # (text, inside main/article)
        self.links: list[tuple[str, str]] = []
        self._link_index: dict[str, int] = {}
        self.images: list[tuple[str, str]] = []
        self._image_index: dict[str, int] = {}
        self._buf: list[str] = []
        self._skip = 0        # inside script/style/…
        self._chrome = 0      # inside nav/footer/… or a boilerplate-classed element
        self._main = 0
        self._in_title = False
        self._pre = 0
        self._head = ""       # heading prefix for the current block
        self._li = 0
        self._href = ""
        self._link_text: list[str] = []
        self._chrome_stack: list[bool] = []  # per open tag: did it raise _chrome?
        self._tag_stack: list[str] = []

    # -- helpers
    def _flush(self):
        text = "".join(self._buf)
        self._buf = []
        if self._pre:
            text = text.strip("\n")
        else:
            text = " ".join(text.split())
        if not text:
            return
        if self._head:
            text = f"{self._head} {text}"
        elif self._li:
            text = f"- {text}"
        self.blocks.append((text, self._main > 0))

    def _abs(self, href: str) -> str:
        try:
            return urllib.parse.urljoin(self.base, href)
        except Exception:
            return href

    # -- parser
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self._tag_stack.append(tag)
        chrome = False
        if tag in _CHROME_TAGS:
            chrome = True
        else:
            hint = f"{a.get('id', '')} {a.get('class', '')} {a.get('role', '')}"
            if hint.strip() and _BOILER_RE.search(hint) and tag not in ("a", "span", "b", "i", "em", "strong", "html", "body", "main", "article"):
                chrome = True
        if a.get("aria-hidden") == "true" or a.get("hidden") is not None:
            chrome = True
        self._chrome_stack.append(chrome)
        if chrome:
            self._chrome += 1
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in ("main", "article"):
            self._main += 1
        if tag in _BLOCK_TAGS:
            self._flush()
            if tag in _HEAD_TAGS:
                self._head = _HEAD_TAGS[tag]
            elif tag == "li":
                self._li += 1
            elif tag == "pre":
                self._pre += 1
            elif tag == "blockquote":
                self._buf.append("> ")
        elif tag == "a":
            href = a.get("href") or ""
            if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
                self._href = self._abs(href)
                self._link_text = []
        elif tag == "img":
            # a picture is a thing their eyes can open (look_at takes a URL):
            # named inline by its alt text, numbered i1, i2… with the index
            # at the end — the pinout, the product photo, the diagram
            alt = " ".join((a.get("alt") or a.get("title") or "").split())
            src = a.get("src") or a.get("data-src") or ""
            if src.startswith("data:") or self._skip or self._chrome:
                return
            url = self._abs(src) if src else ""
            ok = url.startswith(("http://", "https://")) and not re.search(r"\b(?:icon|logo|avatar|sprite|pixel|spacer|badge|emoji|tracking)\b", (src + " " + a.get("class", "")).lower())
            if ok:
                n = self._image_index.get(url)
                if n is None:
                    self.images.append((alt[:80] or "image", url))
                    n = len(self.images)
                    self._image_index[url] = n
                self._buf.append(f"(image: {alt or 'untitled'})[i{n}]")
            elif alt:
                self._buf.append(f"(image: {alt})")

    def handle_endtag(self, tag):
        if self._tag_stack and tag in self._tag_stack:
            # pop to the matching open tag, unwinding what those tags raised
            while self._tag_stack:
                t = self._tag_stack.pop()
                if self._chrome_stack.pop():
                    self._chrome -= 1
                if t == tag:
                    break
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if tag == "a" and self._href:
            text = " ".join("".join(self._link_text).split())
            href = self._href
            self._href = ""
            if text and not self._skip and not self._chrome and href.startswith(("http://", "https://")):
                n = self._link_index.get(href)
                if n is None:
                    self.links.append((text[:80], href))
                    n = len(self.links)
                    self._link_index[href] = n
                self._buf.append(f"{text}[{n}]")
            elif text:
                self._buf.append(text)
            return
        if tag in ("main", "article"):
            self._flush()
            self._main = max(0, self._main - 1)
        if tag in _BLOCK_TAGS:
            self._flush()
            if tag in _HEAD_TAGS:
                self._head = ""
            elif tag == "li":
                self._li = max(0, self._li - 1)
            elif tag == "pre":
                self._pre = max(0, self._pre - 1)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._skip or self._chrome:
            return
        if self._href:
            self._link_text.append(data)
            return
        self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def extract(html: str, url: str) -> Page:
    """The page as light markdown. When <main>/<article> holds a real share
    of the words, only that is kept — the menus and footers go."""
    r = _Reader(url)
    try:
        r.feed(html)
        r.close()
    except Exception:
        pass
    page = Page(url=url, title=" ".join(htmlmod.unescape(r.title).split())[:200], links=r.links, images=r.images)
    inside = [t for t, m in r.blocks if m]
    total = sum(len(t) for t, _ in r.blocks) or 1
    if inside and sum(len(t) for t in inside) >= max(1500, 0.3 * total):
        page.blocks = inside
    else:
        page.blocks = [t for t, _ in r.blocks]
    # a run of one-word blocks is a menu that slipped the class check
    cleaned, run = [], []
    for b in page.blocks:
        if len(b.split()) <= 2 and not b.startswith("#"):
            run.append(b)
            continue
        if len(run) <= 3:
            cleaned.extend(run)
        run = []
        cleaned.append(b)
    if len(run) <= 3:
        cleaned.extend(run)
    page.blocks = cleaned
    return page


def _pages(text: str, size: int) -> list[str]:
    """Cut at paragraph breaks near `size` characters."""
    if len(text) <= size:
        return [text]
    out, rest = [], text
    while len(rest) > size:
        cut = rest.rfind("\n\n", size // 2, size)
        if cut < 0:
            cut = rest.rfind("\n", size // 2, size)
        if cut < 0:
            cut = rest.rfind(" ", size // 2, size)
        if cut < 0:
            cut = size
        out.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        out.append(rest)
    return out


def render(page: Page, part: int = 1, find: str = "", size: int | None = None, links_max: int | None = None) -> str:
    """One part of the page, with the link index for the links that appear
    in that part (and the first few others)."""
    size = int(size or getattr(config, "WEB_PAGE_CHARS", 12000) or 12000)
    links_max = int(links_max or getattr(config, "WEB_LINKS_MAX", 40) or 40)
    parts = _pages(page.text, size) or [""]
    n = len(parts)
    note = ""
    if find:
        needle = find.strip().lower()
        hit = next((i for i, p in enumerate(parts) if needle in p.lower()), None)
        if hit is None:
            note = f" — “{find.strip()}” is not on this page"
        else:
            part = hit + 1
            note = f" — “{find.strip()}” is on this part"
    part = max(1, min(int(part or 1), n))
    body = parts[part - 1]
    head = f"# {page.title}" if page.title else f"# {page.url}"
    where = f"{page.url}" + (f" — part {part} of {n}" if n > 1 else "") + note
    more = (f"\n\n(part {part} of {n} — read_web with page={part + 1} for the next, or find=“…” to jump)"
            if part < n else ("\n\n(the end of the page)" if n > 1 else ""))
    shown = sorted({int(m) for m in re.findall(r"\[(\d+)\]", body)})
    order = [i for i in shown if 1 <= i <= len(page.links)]
    for i in range(1, len(page.links) + 1):
        if len(order) >= links_max:
            break
        if i not in order:
            order.append(i)
    index = ""
    if order:
        lines = [f"[{i}] {page.links[i - 1][0]} → {page.links[i - 1][1]}" for i in order[:links_max]]
        left = len(page.links) - len(lines)
        index = "\n\nlinks on this page (read_web opens any):\n" + "\n".join(lines) + (f"\n(…and {left} more)" if left > 0 else "")
    pics = ""
    if page.images:
        shown_i = sorted({int(m) for m in re.findall(r"\[i(\d+)\]", body)})
        order_i = [i for i in shown_i if 1 <= i <= len(page.images)]
        for i in range(1, len(page.images) + 1):
            if len(order_i) >= links_max:
                break
            if i not in order_i:
                order_i.append(i)
        lines = [f"[i{i}] {page.images[i - 1][0]} → {page.images[i - 1][1]}" for i in order_i[:links_max]]
        left = len(page.images) - len(lines)
        pics = "\n\nimages on this page (look_at opens any — your eyes take a URL):\n" + "\n".join(lines) + (f"\n(…and {left} more)" if left > 0 else "")
    return f"{head}\n{where}\n\n{body}{more}{index}{pics}"


def read(url: str, page: int = 1, find: str = "") -> str:
    """The page, or a plain text / JSON body, or a word about what it is.
    Raises WebError when it cannot be reached."""
    final, ctype, body = fetch(url)
    if "pdf" in ctype or final.lower().split("?")[0].endswith(".pdf"):
        return "PDF"
    text = decode(body, ctype)
    is_html = "html" in ctype or "xml" in ctype or re.search(r"<(?:html|body|div|p|h1)\b", text[:2000], re.IGNORECASE)
    if is_html:
        return render(extract(text, final), part=page, find=find)
    if "json" in ctype:
        try:
            text = json.dumps(json.loads(text), indent=1, ensure_ascii=False)
        except Exception:
            pass
    parts = _pages(text, int(getattr(config, "WEB_PAGE_CHARS", 12000) or 12000))
    n = len(parts)
    p = max(1, min(int(page or 1), n))
    return (f"# {final}" + (f" — part {p} of {n}" if n > 1 else "") + "\n\n" + parts[p - 1]
            + (f"\n\n(part {p} of {n} — read_web with page={p + 1} for the next)" if p < n else ""))


# ---- the search ---------------------------------------------------------------

@dataclass
class Hit:
    title: str
    url: str
    snippet: str


_DDG_A_RE = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
_DDG_SNIP_RE = re.compile(r'<(?:a|td|div|span)[^>]+class="result__snippet"[^>]*>(.*?)</(?:a|td|div|span)>', re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip(s: str) -> str:
    return " ".join(htmlmod.unescape(_TAG_RE.sub("", s or "")).split())


def _ddg_url(href: str) -> str:
    """DuckDuckGo wraps results: //duckduckgo.com/l/?uddg=<url>&rut=…"""
    href = htmlmod.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    try:
        q = urllib.parse.urlparse(href)
        if "duckduckgo.com" in q.netloc and q.path.startswith("/l/"):
            u = urllib.parse.parse_qs(q.query).get("uddg", [""])[0]
            if u:
                return u
    except Exception:
        pass
    return href


_DDG_LITE_A_RE = re.compile(r'<a[^>]+class=["\']result-link["\'][^>]+href="([^"]+)"[^>]*>(.*?)</a>|'
                            r'<a[^>]+href="([^"]+)"[^>]+class=["\']result-link["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
_DDG_LITE_SNIP_RE = re.compile(r'<td[^>]+class=["\']result-snippet["\'][^>]*>(.*?)</td>', re.IGNORECASE | re.DOTALL)


def _ddg_challenged(text: str) -> bool:
    low = text.lower()
    return ("anomaly" in low or "challenge" in low or "bots use duckduckgo" in low) and "result" not in low[:20000]


def _ddg_lite(query: str, n: int) -> list[Hit]:
    """lite.duckduckgo.com — the plainest page they serve, a form POST."""
    body = urllib.parse.urlencode({"q": query, "kl": "us-en"}).encode()
    final, ctype, raw = fetch("https://lite.duckduckgo.com/lite/", data=body, headers={
        "Content-Type": "application/x-www-form-urlencoded", "Referer": "https://lite.duckduckgo.com/",
        "Origin": "https://lite.duckduckgo.com"})
    text = decode(raw, ctype)
    if _ddg_challenged(text):
        raise WebError("challenged")
    hits: list[Hit] = []
    seen = set()
    anchors = list(_DDG_LITE_A_RE.finditer(text))
    for i, m in enumerate(anchors):
        href = m.group(1) or m.group(3) or ""
        title = _strip(m.group(2) or m.group(4) or "")
        url = _ddg_url(href)
        segment = text[m.end():anchors[i + 1].start() if i + 1 < len(anchors) else len(text)]
        sm = _DDG_LITE_SNIP_RE.search(segment)
        snippet = _strip(sm.group(1)) if sm else ""
        if not url.startswith("http") or url in seen or "duckduckgo.com/y.js" in url:
            continue
        seen.add(url)
        hits.append(Hit(title or url, url, snippet))
        if len(hits) >= n:
            break
    return hits


def search_duckduckgo(query: str, n: int) -> list[Hit]:
    """The lite page first (a form POST, the plainest thing they serve),
    then the html page (a POST too); a human check on both is said plainly."""
    try:
        hits = _ddg_lite(query, n)
        if hits:
            return hits
    except WebError:
        pass
    body = urllib.parse.urlencode({"q": query, "kl": "us-en", "b": ""}).encode()
    final, ctype, raw = fetch("https://html.duckduckgo.com/html/", data=body, headers={
        "Content-Type": "application/x-www-form-urlencoded", "Referer": "https://html.duckduckgo.com/",
        "Origin": "https://html.duckduckgo.com"})
    text = decode(raw, ctype)
    if _ddg_challenged(text) or ("anomaly" in text.lower() and "result__a" not in text):
        raise WebError("DuckDuckGo asked for a human check — try again in a while, or set up SearXNG/Brave")
    hits: list[Hit] = []
    seen = set()
    anchors = list(_DDG_A_RE.finditer(text))
    for i, m in enumerate(anchors):
        url = _ddg_url(m.group(1))
        title = _strip(m.group(2))
        # the snippet sits between this result's title and the next result's
        segment = text[m.end():anchors[i + 1].start() if i + 1 < len(anchors) else len(text)]
        sm = _DDG_SNIP_RE.search(segment)
        snippet = _strip(sm.group(1)) if sm else ""
        if not url.startswith("http") or url in seen or "duckduckgo.com/y.js" in url:
            continue
        seen.add(url)
        hits.append(Hit(title or url, url, snippet))
        if len(hits) >= n:
            break
    return hits


def search_searxng(query: str, n: int) -> list[Hit]:
    base = (getattr(config, "WEB_SEARCH_SEARXNG_URL", "") or "").rstrip("/")
    if not base:
        raise WebError("WEB_SEARCH_SEARXNG_URL is not set")
    q = urllib.parse.urlencode({"q": query, "format": "json", "language": "en"})
    final, ctype, body = fetch(f"{base}/search?{q}")
    data = json.loads(decode(body, ctype))
    return [Hit(r.get("title") or r.get("url", ""), r.get("url", ""), " ".join((r.get("content") or "").split()))
            for r in (data.get("results") or [])[:n] if r.get("url")]


def _secret(name: str) -> str:
    """A key from memory/web_search.json — a file that never leaves the
    folder and is not part of the template; never from config."""
    p = config.MEMORY_DIR / "web_search.json"
    try:
        return str((json.loads(p.read_text(encoding="utf-8")) or {}).get(name) or "")
    except Exception:
        return ""


def search_brave(query: str, n: int) -> list[Hit]:
    key = _secret("brave_key")
    if not key:
        raise WebError("no Brave key in memory/web_search.json ({\"brave_key\": \"…\"})")
    q = urllib.parse.urlencode({"q": query, "count": n})
    req = urllib.request.Request(f"https://api.search.brave.com/res/v1/web/search?{q}", headers={
        "Accept": "application/json", "X-Subscription-Token": key, "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raise WebError(f"Brave: HTTP {e.code}") from e
    except Exception as e:
        raise WebError(f"Brave: {e}") from e
    return [Hit(r.get("title", ""), r.get("url", ""), _strip(r.get("description", "")))
            for r in ((data.get("web") or {}).get("results") or [])[:n] if r.get("url")]


ENGINES = {"duckduckgo": search_duckduckgo, "searxng": search_searxng, "brave": search_brave}


def search(query: str, n: int = 8) -> list[Hit]:
    """The configured engine (WEB_SEARCH), falling back to DuckDuckGo."""
    name = (getattr(config, "WEB_SEARCH", "duckduckgo") or "duckduckgo").lower()
    if name not in ENGINES:
        name = "duckduckgo"
    try:
        return ENGINES[name](query, n)
    except WebError:
        if name != "duckduckgo":
            return search_duckduckgo(query, n)
        raise


def render_hits(query: str, hits: list[Hit]) -> str:
    if not hits:
        return f"(searched the web for “{query}” — nothing came back; try other words, or search_wikipedia)"
    out = [f"the web, searching for “{query}” — {len(hits)} result(s); read_web opens any of them:"]
    for i, h in enumerate(hits, 1):
        out.append(f"\n## {i}. {h.title}\n{h.snippet or '(no summary)'}\n{h.url}")
    return "\n".join(out)


# ---- a terminal, for the keeper -------------------------------------------------

def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("search", "read"):
        print(__doc__)
        return
    if sys.argv[1] == "search":
        try:
            print(render_hits(sys.argv[2], search(sys.argv[2], 8)))
        except WebError as e:
            print(f"(search failed: {e})")
        return
    page = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 1
    try:
        out = read(sys.argv[2], page=page)
    except WebError as e:
        print(f"(couldn't reach {sys.argv[2]}: {e})")
        return
    print("(a PDF — read_pdf opens it)" if out == "PDF" else out)


if __name__ == "__main__":
    main()
