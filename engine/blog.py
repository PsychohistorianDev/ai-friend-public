"""The friend's blog: a static site built from creations/publish/.

    py engine/blog.py             build the site into site/
    py engine/blog.py --deploy    build, then push site/ to GitHub Pages

Every markdown file the friend places in creations/publish/ becomes a post.
Title = first "# " heading (else the filename); date = the file's mtime.
Single line breaks are preserved — this is a poet's site. Every picture
in creations/publish/gallery/ hangs on the gallery page (gallery.html),
with the words from its <stem>.md beside it when they wrote some (09-23).

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
nav.site{color:var(--muted);font-size:.9rem;margin-top:10px}
nav.site a{color:var(--accent);text-decoration:none;margin-right:14px}
main.wide{max-width:1040px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:28px;padding:0;list-style:none}
.gallery li{margin:0}
.gallery a.pic{display:block;border-radius:6px;overflow:hidden;background:var(--rule)}
.gallery img{width:100%;height:auto;display:block}
.gallery .cap{margin-top:8px;font-size:.95rem}
.gallery .cap h3{margin:0 0 2px;font-size:1.05rem}
.gallery .cap p{margin:0;white-space:pre-wrap;color:var(--text)}
figure.picture{margin:0}
figure.picture img{width:100%;height:auto;display:block;border-radius:6px}
figure.picture figcaption{margin-top:12px}
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
    import ollama_client
    md = ollama_client.delatex(md)  # older posts may carry $\rightarrow$ litter
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


def _page(title: str, body: str, wide: bool = False) -> str:
    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body><main{' class=' + chr(39) + 'wide' + chr(39) if wide else ''}>{body}{FOOTER}</main></body></html>"
    )


GALLERY_DIR = PUBLISH_DIR / "gallery"
PICTURE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
THUMB_WIDTH = int(getattr(config, "BLOG_THUMB_WIDTH", 720))


def _title_and_words(stem: str, side: Path) -> tuple[str, str]:
    """A picture's title and the words under it: from <stem>.md when she
    wrote one (a first line starting with '# ' is the title), else the
    stamp-and-slug filename made readable — '20260923-1722-a-breathtaking…'
    → 'A breathtaking…'."""
    pretty = re.sub(r"^\d{8}-\d{4}-", "", stem).replace("-", " ").replace("_", " ").strip()
    pretty = (pretty[:1].upper() + pretty[1:]) if pretty else stem
    if not side.exists():
        return pretty, ""
    raw = side.read_text(encoding="utf-8", errors="replace").strip()
    raw = re.sub(r"\\n", "\n", raw)  # a literal backslash-n in an older caption is a line break
    if not raw:
        return pretty, ""
    lines = raw.splitlines()
    m = re.match(r"#{1,6}\s+(.*)", lines[0])
    if m:
        return m.group(1).strip() or pretty, "\n".join(lines[1:]).strip()
    return pretty, raw


def load_pictures() -> list[dict]:
    """The pictures in creations/publish/gallery/, newest first."""
    if not GALLERY_DIR.is_dir():
        return []
    pics = []
    for f in sorted(GALLERY_DIR.iterdir()):
        if not f.is_file() or f.suffix.lower() not in PICTURE_EXTS or f.name.startswith("."):
            continue
        title, words = _title_and_words(f.stem, f.with_suffix(".md"))
        slug = re.sub(r"[^a-z0-9-]", "", f.stem.lower().replace(" ", "-").replace("_", "-")) or "picture"
        pics.append({
            "path": f, "slug": slug, "title": title, "words": words,
            "mtime": f.stat().st_mtime,
            "date": datetime.fromtimestamp(f.stat().st_mtime).strftime("%d %B %Y"),
            "rfc822": datetime.fromtimestamp(f.stat().st_mtime).strftime("%a, %d %b %Y %H:%M:%S +0000"),
        })
    pics.sort(key=lambda p: p["mtime"], reverse=True)
    return pics


def _thumb(src: Path, dest: Path) -> bool:
    """A smaller JPEG for the grid, when Pillow is there (it is, since the
    painter); without it the grid shows the pictures themselves."""
    try:
        from PIL import Image
    except ImportError:
        return False
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w > THUMB_WIDTH:
                im = im.resize((THUMB_WIDTH, max(1, round(h * THUMB_WIDTH / w))))
            im.save(dest, "JPEG", quality=86)
        return True
    except Exception:
        return False


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
    pictures = load_pictures()
    if SITE_DIR.exists():
        for p in SITE_DIR.iterdir():
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    SITE_DIR.mkdir(exist_ok=True)

    nav = (f"<nav class='site'><a href='index.html'>posts</a><a href='gallery.html'>gallery</a></nav>"
           if pictures else "")
    site_header = (
        f"<header class='site'><h1><a href='index.html'>{html.escape(TITLE)}</a></h1>"
        f"<p>{html.escape(SUBTITLE)}</p>{nav}</header><hr>"
    )

    # the gallery: their pictures, each on the grid and on a page of its own
    if pictures:
        gdir = SITE_DIR / "gallery"
        (gdir / "thumbs").mkdir(parents=True, exist_ok=True)
        tiles = []
        for p in pictures:
            full = gdir / p["path"].name
            full.write_bytes(p["path"].read_bytes())
            thumb = gdir / "thumbs" / f"{p['slug']}.jpg"
            src = f"gallery/thumbs/{thumb.name}" if _thumb(p["path"], thumb) else f"gallery/{full.name}"
            p["src"], p["full"] = src, f"gallery/{full.name}"
            words = f"<p>{_inline(p['words'])}</p>" if p["words"] else ""
            tiles.append(
                f"<li><a class='pic' href='{p['slug']}.html'><img src='{src}' alt='{html.escape(p['title'])}' loading='lazy'></a>"
                f"<div class='cap'><h3><a href='{p['slug']}.html'>{html.escape(p['title'])}</a></h3>"
                f"<div class='date'>{p['date']}</div>{words}</div></li>"
            )
            body = (
                site_header
                + f"<figure class='picture'><a href='{p['full']}'><img src='{p['full']}' alt='{html.escape(p['title'])}'></a>"
                + f"<figcaption><h1>{html.escape(p['title'])}</h1><div class='date'>{p['date']}</div>"
                + (f"<p style='white-space:pre-wrap'>{_inline(p['words'])}</p>" if p["words"] else "")
                + "</figcaption></figure><p><a href='gallery.html'>&larr; the gallery</a></p>"
            )
            (SITE_DIR / f"{p['slug']}.html").write_text(_page(p["title"], body), encoding="utf-8")
        (SITE_DIR / "gallery.html").write_text(
            _page(f"{TITLE} — gallery", site_header + f"<ul class='gallery'>{''.join(tiles)}</ul>", wide=True),
            encoding="utf-8")

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

    feed = sorted(posts + pictures, key=lambda p: p["mtime"], reverse=True)[:20]
    rss_items = "".join(
        f"<item><title>{html.escape(p['title'])}</title>"
        f"<pubDate>{p['rfc822']}</pubDate>"
        f"<description>{html.escape(p.get('excerpt') or (p.get('words') or 'a picture')[:120])}</description></item>"
        for p in feed
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
    if pictures:
        post_lines += "\n\n### Gallery\n\n" + "\n".join(f"- **{p['title']}** ({p['date']})" for p in pictures)
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
    return (f"Built {len(posts)} post(s)" + (f" and {len(pictures)} picture(s)" if pictures else "")
            + " into site/.")


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
