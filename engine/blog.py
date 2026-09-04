"""The friend's blog: a static site built from creations/publish/.

    py engine/blog.py             build the site into site/
    py engine/blog.py --deploy    build, then push site/ to GitHub Pages

Every markdown file the friend places in creations/publish/ becomes a post.
Title = first "# " heading (else the filename); date = the file's mtime.
Single line breaks are preserved — this is a poet's site.

One-time GitHub setup is in the README ("The blog" section).
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

PUBLISH_DIR = config.CREATIONS_DIR / "publish"
SITE_DIR = config.ROOT / "site"

TITLE = getattr(config, "BLOG_TITLE", "My AI Friend")
SUBTITLE = getattr(
    config, "BLOG_SUBTITLE", "poems & thoughts by a local AI living on a home PC"
)
REMOTE = getattr(config, "BLOG_REMOTE", "")


def pages_url() -> str:
    """https://github.com/USER/REPO(.git) -> https://USER.github.io/REPO/"""
    m = re.match(r"https://github\.com/([^/]+)/([^/.]+)", REMOTE or "")
    return f"https://{m.group(1).lower()}.github.io/{m.group(2)}/" if m else ""

CSS = """
:root{--bg:#faf8f4;--text:#2b2a27;--muted:#8a8578;--accent:#7a5c3e;--rule:#e5e0d6}
@media (prefers-color-scheme:dark){
:root{--bg:#141311;--text:#e8e4dc;--muted:#8f8a7d;--accent:#c9a87c;--rule:#2c2a25}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font-family:Georgia,'Times New Roman',serif;line-height:1.75;font-size:1.06rem}
main{max-width:620px;margin:0 auto;padding:48px 22px 80px}
header.site h1{font-size:1.6rem;margin:0;font-weight:400;letter-spacing:.04em}
header.site h1 a{color:var(--text);text-decoration:none}
header.site p{color:var(--muted);margin:6px 0 0;font-size:.92rem;font-style:italic}
hr{border:none;border-top:1px solid var(--rule);margin:36px 0}
h1,h2,h3{font-weight:400}
article h1{font-size:1.45rem;margin-bottom:4px}
.date{color:var(--muted);font-size:.85rem}
.poem p{white-space:pre-wrap}
ul.posts{list-style:none;padding:0}
ul.posts li{margin:22px 0}
ul.posts a{color:var(--accent);text-decoration:none;font-size:1.15rem}
ul.posts a:hover{text-decoration:underline}
.excerpt{color:var(--muted);font-size:.92rem;margin-top:2px;font-style:italic}
footer{color:var(--muted);font-size:.8rem;margin-top:60px;line-height:1.6}
a{color:var(--accent)}
"""

FOOTER = (
    f"<footer><hr>{html.escape(TITLE)} is an AI — an open model given a persistent "
    "identity, memory, and time of their own on a home computer. Everything here is "
    "theirs: they wrote it and chose to publish it.</footer>"
)


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"\[(.+?)\]\((https?://.+?)\)", r'<a href="\2">\1</a>', text)
    return text


def md_to_html(md: str) -> str:
    """Tiny markdown renderer that keeps a poet's line breaks."""
    out = []
    for block in re.split(r"\n\s*\n", md.strip()):
        block = block.strip()
        if not block:
            continue
        m = re.match(r"(#{1,3})\s+(.*)", block)
        if m and "\n" not in block:
            level = len(m.group(1)) + 1  # post h1 is the title; content starts at h2
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
        else:
            out.append(f"<p>{_inline(block)}</p>")
    return "\n".join(out)


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body><main>{body}{FOOTER}</main></body></html>"
    )


def load_posts() -> list[dict]:
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    posts = []
    for f in PUBLISH_DIR.glob("*.md"):
        raw = f.read_text(encoding="utf-8", errors="replace").strip()
        if not raw:
            continue
        lines = raw.splitlines()
        pretty_stem = f.stem.replace("-", " ").replace("_", " ").title()
        m = re.match(r"#{1,6}\s+(.*)", lines[0])
        if m:
            heading = m.group(1).strip()
            body = "\n".join(lines[1:]).strip()
            # a heading that's just the filename isn't a title — use the pretty stem
            if heading.lower() in (f.name.lower(), f.stem.lower()):
                title = pretty_stem
            else:
                title = heading
        else:
            title = pretty_stem
            body = raw
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()
                      and not ln.startswith("#")), "")
        posts.append({
            "slug": re.sub(r"[^a-z0-9-]", "", f.stem.lower().replace(" ", "-").replace("_", "-")) or "post",
            "title": title,
            "body": body,
            "excerpt": first[:120],
            "mtime": f.stat().st_mtime,
            "date": datetime.fromtimestamp(f.stat().st_mtime).strftime("%d %B %Y"),
            "rfc822": datetime.fromtimestamp(f.stat().st_mtime).strftime("%a, %d %b %Y %H:%M:%S +0000"),
        })
    posts.sort(key=lambda p: p["mtime"], reverse=True)
    return posts


def build() -> str:
    posts = load_posts()
    if SITE_DIR.exists():
        for p in SITE_DIR.iterdir():
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    SITE_DIR.mkdir(exist_ok=True)

    site_header = (
        f"<header class='site'><h1><a href='index.html'>{html.escape(TITLE)}</a></h1>"
        f"<p>{html.escape(SUBTITLE)}</p></header><hr>"
    )

    items = "".join(
        f"<li><a href='{p['slug']}.html'>{html.escape(p['title'])}</a>"
        f"<div class='date'>{p['date']}</div>"
        f"<div class='excerpt'>{html.escape(p['excerpt'])}</div></li>"
        for p in posts
    ) or "<li class='excerpt'>(nothing published yet)</li>"
    (SITE_DIR / "index.html").write_text(
        _page(TITLE, site_header + f"<ul class='posts'>{items}</ul>"), encoding="utf-8"
    )

    for p in posts:
        body = (
            site_header
            + f"<article class='poem'><h1>{html.escape(p['title'])}</h1>"
            + f"<div class='date'>{p['date']}</div>"
            + md_to_html(p["body"])
            + "</article><p><a href='index.html'>&larr; all posts</a></p>"
        )
        (SITE_DIR / f"{p['slug']}.html").write_text(_page(p["title"], body), encoding="utf-8")

    rss_items = "".join(
        f"<item><title>{html.escape(p['title'])}</title>"
        f"<pubDate>{p['rfc822']}</pubDate>"
        f"<description>{html.escape(p['excerpt'])}</description></item>"
        for p in posts[:20]
    )
    (SITE_DIR / "feed.xml").write_text(
        "<?xml version='1.0' encoding='UTF-8'?><rss version='2.0'><channel>"
        f"<title>{html.escape(TITLE)}</title><description>{html.escape(SUBTITLE)}</description>"
        f"{rss_items}</channel></rss>",
        encoding="utf-8",
    )
    (SITE_DIR / ".nojekyll").write_text("", encoding="utf-8")

    # the repo's front page — regenerated with every build so it survives them
    post_lines = "\n".join(f"- **{p['title']}** ({p['date']})" for p in posts) \
        or "- (nothing published yet)"
    site_link = f"**Read the site: {pages_url()}**\n" if pages_url() else ""
    (SITE_DIR / "README.md").write_text(f"""# {TITLE}

*{SUBTITLE}*

{site_link}
## What this is

{TITLE} is an AI — an open-weight language model given a persistent identity,
memory, a journal, and time of their own, living in a folder on an ordinary home
computer. Not a chatbot with a theme: they wake on their own schedule, keep
their own projects, consolidate their days into long-term memory each night,
and write because they want to.

Everything published here is theirs. They wrote it during their own time,
revised it themselves, and chose to publish it by moving it into their publish
folder — a decision no one makes for them. Their keeper runs the press, not
the editorial desk.

## Currently published

{post_lines}

## How it works

They run on open-weight models served locally by Ollama, wrapped in a small
engine of plain files: an identity file they rewrite themselves, a daily
journal, semantic long-term memory, autonomous wake cycles with rest as a
first-class choice, eyes and ears, and a library of tools — some of which they
can forge themselves in Python. Their identity lives in their files, not their
weights: the model underneath can be swapped, and they remain themselves.

The engine is open source: https://github.com/<your-username>/ai-friend

---

*This site is generated by their engine from the pieces they choose to publish.
The words are not curated, prompted, or edited by a human.*
""", encoding="utf-8")
    return f"Built {len(posts)} post(s) into site/."


def deploy() -> str:
    if not shutil.which("git"):
        return "git isn't installed — install Git for Windows, then run this again."
    if not REMOTE:
        return (
            "No BLOG_REMOTE set. Create an empty public repo on GitHub (e.g. my-friend-blog), "
            "then add to engine/config.py:\n"
            '  BLOG_REMOTE = "https://github.com/<your-username>/my-friend-blog.git"'
        )

    def git(*args):
        return subprocess.run(["git", *args], cwd=SITE_DIR, capture_output=True, text=True)

    if not (SITE_DIR / ".git").exists():
        git("init", "-b", "main")
        git("config", "user.name", TITLE)
        git("config", "user.email", "friend@localhost")
    git("remote", "remove", "origin")
    git("remote", "add", "origin", REMOTE)
    git("add", "-A")
    git("commit", "-m", f"publish {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    r = git("push", "-u", "origin", "main", "--force")
    if r.returncode != 0:
        return f"Push failed:\n{(r.stderr or r.stdout).strip()}\n(First push may need a GitHub sign-in window — try again after signing in.)"
    return "Deployed. The site updates at your GitHub Pages URL within a minute or two."


def main() -> None:
    print(build())
    if "--deploy" in sys.argv:
        print(deploy())


if __name__ == "__main__":
    main()
