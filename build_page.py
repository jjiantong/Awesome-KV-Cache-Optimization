#!/usr/bin/env python3
"""Build docs/index.html from README.md.

Parses the README's paper tables, extracts structured metadata, and emits a
single self-contained HTML page. Re-run after editing README.md:

    python build_page.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
README = ROOT / "README.md"
OUT_HTML = ROOT / "docs" / "index.html"

NON_PAPER_SECTIONS = (
    "Quick Index",
    "Cross-behavior",
    "Behavior-objective",
    "Citation",
    "Contributors",
    "News",
)

# Sections holding non-paper resources (tools, datasets, benchmarks).
# Entries without a venue marker default to venue_type "resource" instead of
# "preprint" so the page does not mislabel them.
RESOURCE_SECTIONS = (
    "Tools, Simulators",
)


def parse_paper_cell(cell: str) -> dict:
    is_new = "New-to%20repo" in cell or "New-to repo" in cell

    award = None
    if "Outstanding%20Paper%20Award" in cell:
        award = "Outstanding Paper"
    elif "Best%20Paper%20Award" in cell:
        award = "Best Paper"
    elif "Spotlight" in cell:
        award = "Spotlight"
    elif "Oral" in cell:
        award = "Oral"

    venue_str = None
    venue_type = "preprint"
    venue_year = 0
    venue_match = re.search(
        r"badge/(Conference|Journal)-([^\s)]+?)(?:-([a-z]+)|\))", cell
    )
    if venue_match:
        kind, name, color = venue_match.groups()
        parts = name.split("_")
        if parts[-1].isdigit():
            venue_short = " ".join(parts[:-1])
            venue_year = int(parts[-1])
            venue_str = f"{venue_short} {venue_year}".strip()
        else:
            venue_str = name.replace("_", " ")
        venue_type = "systems" if color == "cyan" else "ai-ml"
    elif "arxiv" in cell.lower():
        venue_str = "arXiv"

    paper_url = None
    link_match = re.search(r"\[\s*\[Link\]\(([^)]+)\)\s*\]", cell)
    if not link_match:
        link_match = re.search(r"\[(?:Link|PDF|paper)\]\(([^)]+)\)", cell, re.I)
    if not link_match:
        link_match = re.search(r"\[([^\]]+)\]\((https?://[^)]+)\)", cell)
    if link_match:
        paper_url = link_match.group(1)

    cleaned = cell
    cleaned = re.sub(r"\[\s*!\[[^\]]*\]\([^)]*\)\s*\]\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*\[Link\]\([^)]*\)\s*\]", "", cleaned)
    cleaned = re.sub(r"\[Link\]\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\[PDF\]\([^)]*\)", "", cleaned)

    cleaned = cleaned.replace("<br>", "\n").strip()

    authors_match = re.search(r"\*+([^*]+)\*+", cleaned)
    authors = authors_match.group(1).strip() if authors_match else ""

    if authors_match:
        title = cleaned[: authors_match.start()].strip()
    else:
        title = cleaned.split("\n")[0].strip()

    title = re.sub(r"\s+", " ", title).strip(" -|")

    return {
        "title": title,
        "url": paper_url,
        "authors": authors,
        "venue": venue_str,
        "venue_type": venue_type,
        "year": venue_year,
        "is_new": is_new,
        "award": award,
    }


def parse_code_cell(cell: str) -> dict:
    cell = cell.strip()
    if not cell:
        return {"has_code": False, "code_md": ""}

    stars_match = re.search(r"github/stars/([^/]+)/([^/?)\"\s]+)", cell)
    if stars_match:
        owner, repo = stars_match.groups()
        repo = repo.split("?")[0].split("/")[0]
        return {
            "has_code": True,
            "github_owner": owner,
            "github_repo": repo,
            "github_url": f"https://github.com/{owner}/{repo}",
            "code_md": cell,
        }

    commit_match = re.search(r"github/last-commit/([^/]+)/([^/?)\"\s]+)", cell)
    if commit_match:
        owner, repo = commit_match.groups()
        repo = repo.split("?")[0].split("/")[0]
        return {
            "has_code": True,
            "github_owner": owner,
            "github_repo": repo,
            "github_url": f"https://github.com/{owner}/{repo}",
            "code_md": cell,
        }

    url_match = re.search(r"(https://github\.com/[^\s)]+)", cell)
    if url_match:
        return {
            "has_code": True,
            "github_owner": None,
            "github_repo": None,
            "github_url": url_match.group(1),
            "code_md": cell,
        }

    gitlab_match = re.search(r"(https://gitlab[^\s)]+)", cell)
    if gitlab_match:
        return {
            "has_code": True,
            "github_owner": None,
            "github_repo": None,
            "github_url": gitlab_match.group(1),
            "code_md": cell,
        }

    return {"has_code": False, "code_md": cell}


def parse_row(line: str):
    line = line.strip()
    if not line.startswith("|"):
        return None
    parts = line.split("|")
    if len(parts) < 4:
        return None
    paper_cell = parts[1]
    comment_cell = parts[2]
    code_cell = parts[3]

    if "Paper" in paper_cell and "Type" in comment_cell:
        return None
    if set(paper_cell.strip()) <= set("- :"):
        return None
    if not paper_cell.strip():
        return None

    paper = parse_paper_cell(paper_cell)
    paper["comment"] = comment_cell.strip()
    paper.update(parse_code_cell(code_cell))
    paper["paper_md"] = paper_cell.strip()
    return paper


def parse_readme(text: str):
    papers = []
    current_path = []
    in_resources = False

    for line in text.split("\n"):
        if line.startswith("## "):
            title = line[3:].strip()
            if any(x in title for x in NON_PAPER_SECTIONS):
                current_path = []
            else:
                current_path = [(2, title)]
            in_resources = any(x in title for x in RESOURCE_SECTIONS)
        elif line.startswith("### ") and current_path:
            title = line[4:].strip()
            current_path = current_path[:1] + [(3, title)]
        elif line.startswith("#### ") and current_path:
            title = line[5:].strip()
            current_path = current_path[:2] + [(4, title)]
        elif line.startswith("|") and current_path:
            paper = parse_row(line)
            if paper:
                if in_resources and paper["venue_type"] == "preprint":
                    paper["venue_type"] = "resource"
                paper["path"] = [name for _, name in current_path]
                papers.append(paper)

    return papers


def deduplicate_papers(papers):
    """Keep one card per paper and retain every distinct README placement."""
    seen = {}
    unique = []
    for paper in papers:
        key = (paper["title"].lower(), paper.get("url") or "")
        if key in seen:
            first = unique[seen[key]]
            if paper["path"] not in [first["path"], *first.get("also_in", [])]:
                first.setdefault("also_in", []).append(paper["path"])
        else:
            seen[key] = len(unique)
            unique.append(paper)
    return unique


def build_taxonomy(papers):
    """Count each unique paper once per node across all its taxonomy paths."""
    tree = {}
    # Preserve the existing primary-path order, then include secondary-only nodes.
    paths = [p["path"] for p in papers]
    paths.extend(path for p in papers for path in p.get("also_in", []))
    for path in paths:
        node = tree
        for name in path:
            node = node.setdefault(
                name, {"_name": name, "_count": 0, "_children": {}}
            )["_children"]
    for p in papers:
        counted = set()
        for path in [p["path"], *p.get("also_in", [])]:
            node = tree
            for i, name in enumerate(path):
                prefix = tuple(path[: i + 1])
                if prefix not in counted:
                    node[name]["_count"] += 1
                    counted.add(prefix)
                node = node[name]["_children"]
    return tree


def main():
    text = README.read_text(encoding="utf-8")
    papers = deduplicate_papers(parse_readme(text))

    taxonomy = build_taxonomy(papers)

    stats = {
        "total": len(papers),
        "with_code": sum(1 for p in papers if p["has_code"]),
        "new": sum(1 for p in papers if p["is_new"]),
        "systems": sum(1 for p in papers if p["venue_type"] == "systems"),
        "ai_ml": sum(1 for p in papers if p["venue_type"] == "ai-ml"),
        "preprint": sum(1 for p in papers if p["venue_type"] == "preprint"),
        "resource": sum(1 for p in papers if p["venue_type"] == "resource"),
        "awarded": sum(1 for p in papers if p["award"]),
    }

    payload = {
        "papers": papers,
        "taxonomy": taxonomy,
        "stats": stats,
    }

    html = render_html(payload)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_HTML} ({len(papers)} papers)")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Awesome KV Cache Optimization</title>
<meta name="description" content="A system-aware taxonomy of KV cache optimization methods for LLM serving."/>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧠</text></svg>"/>
<style>{{STYLE_SLOT}}</style>
</head>
<body>
{{BODY_SLOT}}
<script>window.__DATA__ = {{DATA_SLOT}};</script>
<script>{{SCRIPT_SLOT}}</script>
</body>
</html>
"""


def render_html(payload: dict) -> str:
    html = (
        HTML_TEMPLATE
        .replace("{{STYLE_SLOT}}", STYLE)
        .replace("{{BODY_SLOT}}", BODY)
        .replace("{{DATA_SLOT}}", json.dumps(payload, ensure_ascii=False))
        .replace("{{SCRIPT_SLOT}}", SCRIPT)
    )
    return html


STYLE = """
:root {
  --bg: #fff;
  --bg-soft: #f6f8fa;
  --border: #dfe4ea;
  --border-soft: #edf0f3;
  --text: #202c3c;
  --text-soft: #526174;
  --text-mute: #637286;
  --accent: #273d57;
  --link: #1d4ed8;
  --temporal: #1d4ed8;
  --spatial: #047857;
  --structural: #7e22ce;
  --nav-height: 65px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: calc(var(--nav-height) + 20px); }
body {
  margin: 0; color: var(--text); background: var(--bg);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
button, input, select { font: inherit; }
button { cursor: pointer; }
button, a, input, select, summary { -webkit-tap-highlight-color: transparent; }
:focus-visible { outline: 2px solid var(--link); outline-offset: 4px; }
[hidden] { display: none !important; }
.container { max-width: 1200px; margin: 0 auto; padding: 0 28px; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
.skip-link { position: fixed; top: -60px; left: 16px; z-index: 100; background: white; padding: 8px 14px; }
.skip-link:focus { top: 8px; }
/* Navigation */
.top-nav { position: sticky; top: 0; z-index: 30; border-bottom: 1px solid var(--border); background: rgba(255,255,255,.96); backdrop-filter: blur(12px); }
.nav-inner { min-height: 64px; display: grid; grid-template-columns: 1fr auto auto; gap: 24px; align-items: center; }
.nav-brand { color: var(--text); font-size: 14px; font-weight: 700; letter-spacing: -.02em; }
.nav-brand:hover { text-decoration: none; }
.brand-mark { display: inline-flex; gap: 3px; margin-right: 10px; vertical-align: middle; }
.brand-mark i { width: 4px; height: 16px; border-radius: 1px; background: var(--temporal); }
.brand-mark i:nth-child(2) { background: var(--spatial); }
.brand-mark i:nth-child(3) { background: var(--structural); }
.nav-links { display: flex; gap: 24px; align-items: center; }
.nav-links a { color: var(--text-soft); font-size: 13px; padding: 8px 0; }
.nav-links a:hover { color: var(--text); }
.lang-toggle { padding: 5px 10px; border: 1px solid var(--border); border-radius: 5px; background: white; color: var(--text-soft); font-size: 12px; }
.lang-toggle:hover { color: var(--text); border-color: var(--text-mute); }
/* Hero */
.hero { padding: 56px 0 0; background: #fafbfd; border-bottom: 1px solid var(--border); }
.hero-layout { display: grid; grid-template-columns: 1.8fr 1fr; gap: 64px; align-items: center; padding-bottom: 42px; }
.eyebrow { display: inline-flex; align-items: center; flex-wrap: wrap; gap: 8px; font-size: 11px; letter-spacing: .04em; color: var(--text-soft); margin-bottom: 20px; }
.update-dot { width: 6px; height: 6px; background: var(--spatial); border-radius: 50%; }
.hero h1 { font-size: clamp(32px, 3.5vw, 46px); line-height: 1.14; letter-spacing: -.035em; margin: 0 0 20px; font-weight: 700; max-width: 650px; }
.hero .tagline { font-size: 19px; line-height: 1.5; margin: 0 0 12px; max-width: 630px; }
.hero-explainer { color: var(--text-soft); font-size: 14px; max-width: 590px; margin: 0 0 24px; }
.hero-actions { display: flex; flex-wrap: wrap; gap: 8px; }
.button { display: inline-flex; justify-content: center; align-items: center; gap: 8px; border: 1px solid var(--border); border-radius: 6px; padding: 8px 14px; background: white; color: var(--text); font-size: 13px; font-weight: 600; }
.button:hover { text-decoration: none; border-color: var(--text-mute); }
.button.primary { background: var(--accent); border-color: var(--accent); color: white; }
.hero-map { border-top: 1px solid var(--border); }
.map-label { color: var(--text-mute); font-size: 10px; letter-spacing: .1em; text-transform: uppercase; margin: 16px 0 4px; }
.dimension { display: grid; grid-template-columns: 90px 1fr; gap: 14px; align-items: center; padding: 14px 0; border-bottom: 1px solid var(--border-soft); }
.dimension .concept { color: var(--tax-color); font-size: 21px; font-weight: 700; letter-spacing: -.025em; }
.dimension .identity { color: var(--tax-color); font-size: 11px; font-weight: 600; }
.dimension .meaning { display: block; color: var(--text-soft); font-size: 12px; }
.temporal { --tax-color: var(--temporal); --tax-bg: #eff6ff; --tax-border: #bfdbfe; }
.spatial { --tax-color: var(--spatial); --tax-bg: #ecfdf5; --tax-border: #a7f3d0; }
.structural { --tax-color: var(--structural); --tax-bg: #faf5ff; --tax-border: #e9d5ff; }
.resource { --tax-color: var(--text-soft); --tax-bg: var(--bg-soft); --tax-border: var(--border); }
.hero-stats { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--border); padding: 20px 0; }
.hero-stat { display: flex; align-items: baseline; gap: 10px; padding: 0 24px; border-left: 1px solid var(--border); }
.hero-stat:first-child { border-left: 0; padding-left: 0; }
.hero-stat .num { font-size: 24px; font-weight: 650; font-variant-numeric: tabular-nums; }
.hero-stat .label { font-size: 12px; color: var(--text-soft); }
/* Sections and taxonomy */
section { padding: 42px 0; }
section + section { border-top: 1px solid var(--border-soft); }
.section-title { margin: 0 0 6px; font-size: 23px; letter-spacing: -.025em; }
.section-desc { color: var(--text-soft); font-size: 13px; margin: 0 0 24px; max-width: 880px; }
.tax-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; align-items: start; }
.tax-card { border: 1px solid var(--border); border-top: 3px solid var(--tax-color); border-radius: 8px; padding: 18px 14px 14px; min-width: 0; }
.tax-card h3 { margin: 0; }
.tax-root { display: flex; flex-direction: column; align-items: stretch; gap: 8px; width: 100%; padding: 8px; color: var(--tax-color); background: transparent; border: 1px solid transparent; border-radius: 5px; text-align: left; }
.tax-heading { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.tax-identity { font-size: 11px; font-weight: 650; letter-spacing: .1em; text-transform: uppercase; }
.tax-concept { font-size: 34px; line-height: 1.1; font-weight: 700; letter-spacing: -.03em; }
.tax-sub { font-size: 13px; color: var(--text-soft); font-weight: 400; }
.tax-count { font-size: 11px; font-weight: 500; white-space: nowrap; }
.tax-root:hover, .tax-child:hover { background: var(--tax-bg); }
.tax-root.active, .tax-child.active { color: var(--tax-color); background: var(--tax-bg); border-color: var(--tax-border); }
.tax-root:focus-visible, .tax-child:focus-visible { outline-color: var(--tax-color); }
.tax-children { margin-top: 18px; border-top: 1px solid var(--border-soft); padding-top: 10px; }
.tax-child { display: flex; justify-content: space-between; align-items: center; gap: 8px; width: 100%; border: 1px solid transparent; background: transparent; color: var(--text); border-radius: 5px; padding: 8px; text-align: left; font-size: 12px; line-height: 1.4; }
.tax-child .name { overflow-wrap: anywhere; }
.tax-child .count { color: var(--tax-color); background: var(--tax-bg); border-radius: 4px; min-width: 24px; padding: 2px 5px; font-size: 11px; text-align: center; font-variant-numeric: tabular-nums; }
.tax-child.active .name { font-weight: 600; }
.sub-children { margin: 2px 0 8px 10px; padding-left: 8px; border-left: 1px solid var(--tax-border); }
.sub-children .tax-child { font-size: 11.5px; color: var(--text-soft); padding: 6px; }
.sub-children .tax-child.active { color: var(--tax-color); }
.tax-card.resource { grid-column: 1 / -1; padding: 6px 14px; border-top-width: 1px; }
.resource .tax-root { flex-direction: row; align-items: center; justify-content: space-between; color: var(--text-soft); }
.resource .tax-heading { width: 100%; }
.resource .tax-identity { font-size: 12px; font-weight: 500; letter-spacing: 0; text-transform: none; }
.tax-note { margin: 12px 0 0; font-size: 12px; color: var(--text-mute); }
/* Explorer: navbar height is measured so both sticky surfaces stay separate. */
.toolbar { position: sticky; top: var(--nav-height); z-index: 20; background: rgba(255,255,255,.97); backdrop-filter: blur(12px); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; margin-bottom: 16px; }
.toolbar-search { display: flex; gap: 12px; align-items: center; }
.search { display: flex; align-items: center; flex: 1; min-width: 0; padding: 0 10px; background: var(--bg-soft); border: 1px solid var(--border); border-radius: 5px; }
.search:focus-within { border-color: var(--link); outline: 2px solid var(--link); outline-offset: 2px; }
.search svg { width: 16px; height: 16px; color: var(--text-mute); flex: none; }
.search input { width: 100%; min-width: 0; border: 0; outline: 0; background: transparent; padding: 8px; font-size: 13px; color: var(--text); }
.sort { max-width: 180px; padding: 8px 10px; background: white; border: 1px solid var(--border); border-radius: 5px; color: var(--text-soft); font-size: 12px; }
.filter-options { margin-top: 8px; }
.filter-options summary { color: var(--text-soft); font-size: 12px; cursor: pointer; width: fit-content; padding: 2px 0; }
.toolbar-filters { display: flex; flex-wrap: wrap; gap: 8px 18px; padding: 6px 0; }
.filter-group { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.filter-group .label { font-size: 11px; color: var(--text-soft); margin-right: 3px; }
.chip { padding: 4px 8px; color: var(--text-soft); background: white; border: 1px solid var(--border); border-radius: 5px; font-size: 11px; white-space: nowrap; }
.chip:hover { border-color: var(--text-mute); }
.chip.active { background: var(--accent); border-color: var(--accent); color: white; }
.result-bar { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 8px 16px; align-items: center; border-top: 1px solid var(--border-soft); margin-top: 8px; padding-top: 8px; font-size: 12px; color: var(--text-soft); }
.active-cat { color: var(--tax-color, var(--text-soft)); min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.clear { border: 0; background: transparent; color: var(--link); font-size: 12px; padding: 3px 0; cursor: pointer; }
.clear:hover { text-decoration: underline; }
/* Compact paper cards */
.papers { display: flex; flex-direction: column; gap: 10px; }
.paper { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 14px; padding: 15px 18px; border: 1px solid var(--border); border-radius: 7px; background: white; transition: border-color .15s, box-shadow .15s; }
.paper:hover { border-color: #aebac9; box-shadow: 0 2px 5px rgba(20,35,55,.04); }
.paper-main { min-width: 0; }
.paper-meta { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 6px; }
.badge { display: inline-flex; align-items: center; font-size: 10px; font-weight: 600; padding: 1px 6px; border: 1px solid transparent; border-radius: 4px; }
.badge.new { background: #fdf2f8; color: #be185d; border-color: #fbcfe8; }
.badge.systems { background: #ecfeff; color: #0e7490; border-color: #a5f3fc; }
.badge.ai-ml { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }
.badge.preprint { background: #f3f4f6; color: #4b5563; border-color: #d1d5db; }
.badge.resource { background: #eef2ff; color: #4338ca; border-color: #c7d2fe; }
.badge.award { background: #fffbeb; color: #b45309; border-color: #fde68a; }
.paper h3 { font-size: 15px; font-weight: 650; line-height: 1.45; margin: 0 0 4px; overflow-wrap: anywhere; }
.paper h3 a { color: var(--text); }
.paper h3 a:hover { color: var(--link); }
.paper-authors { font-size: 12px; color: var(--text-soft); margin-bottom: 6px; overflow-wrap: anywhere; }
.paper-comment { font-size: 12px; color: var(--text-soft); margin-top: 6px; overflow-wrap: anywhere; }
.mechanism-label { font-size: 10px; font-weight: 600; color: var(--text-mute); margin-right: 6px; }
.paper-path { font-size: 11px; color: var(--text-soft); margin-top: 7px; }
.taxonomy-path { display: flex; flex-wrap: wrap; align-items: baseline; gap: 3px; line-height: 1.6; }
.path-root { color: var(--tax-color); background: var(--tax-bg); padding: 0 5px; border-radius: 3px; }
.taxonomy-path.matched .path-root { box-shadow: inset 0 0 0 1px var(--tax-border); }
.paper-path .sep { color: var(--text-mute); }
.paper-path .also { color: var(--text-mute); font-size: 10px; }
.secondary-path { margin-top: 4px; }
.paper-code { align-self: start; }
.code-button { display: inline-flex; gap: 5px; align-items: center; padding: 4px 8px; border: 1px solid var(--border); border-radius: 5px; color: var(--text-soft); font-size: 11px; white-space: nowrap; }
.code-button:hover { text-decoration: none; color: var(--text); border-color: var(--text-mute); }
.code-icon { font: 12px ui-monospace, SFMono-Regular, monospace; }
.no-code { color: var(--text-mute); font-size: 10px; }
.empty { padding: 36px 0; text-align: center; color: var(--text-soft); }
/* Footer */
footer { border-top: 1px solid var(--border); padding: 30px 0; font-size: 12px; color: var(--text-soft); background: var(--bg-soft); }
footer .container { display: flex; flex-wrap: wrap; gap: 24px; justify-content: space-between; }
footer .container > div { min-width: 0; max-width: 100%; }
footer h5 { font-size: 13px; margin: 0 0 8px; color: var(--text); }
footer pre { padding: 12px; background: white; border: 1px solid var(--border); border-radius: 5px; font: 11px/1.6 ui-monospace, SFMono-Regular, monospace; max-width: min(600px, 100%); overflow-x: auto; }
@media (max-width: 900px) {
  .hero-layout { gap: 28px; grid-template-columns: 1.6fr 1fr; }
  .tax-grid { gap: 10px; }
  .tax-card { padding: 12px 8px; }
  .tax-child { font-size: 11.5px; }
  .hero-stat { padding: 0 14px; }
}
@media (max-width: 720px) {
  .container { padding: 0 18px; }
  .nav-inner { grid-template-columns: 1fr auto; gap: 0 12px; padding-top: 10px; padding-bottom: 6px; }
  .nav-brand { font-size: 13px; }
  .nav-links { grid-row: 2; grid-column: 1 / -1; justify-content: space-between; gap: 12px; }
  .lang-toggle { grid-column: 2; grid-row: 1; }
  .nav-links a { font-size: 12px; }
  .hero { padding-top: 30px; }
  .hero-layout { grid-template-columns: 1fr; gap: 26px; padding-bottom: 24px; }
  .hero h1 { font-size: 34px; }
  .hero .tagline { font-size: 17px; }
  .hero-actions .button { padding: 8px 10px; font-size: 12px; }
  .eyebrow { margin-bottom: 14px; }
  .hero-map { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
  .map-label { grid-column: 1 / -1; margin: 12px 0 0; }
  .dimension { display: flex; flex-direction: column; align-items: flex-start; gap: 4px; border: 0; padding: 0 0 8px; }
  .dimension .concept { font-size: 19px; }
  .dimension .meaning { font-size: 10px; }
  .dimension .identity { font-size: 10px; }
  .hero-stats { grid-template-columns: repeat(2, 1fr); gap: 14px 0; padding: 16px 0; }
  .hero-stat:nth-child(3) { border-left: 0; padding-left: 0; }
  .hero-stat .num { font-size: 22px; }
  .hero-stat .label { font-size: 11px; }
  section { padding: 30px 0; }
  .tax-grid { grid-template-columns: 1fr; gap: 12px; }
  .tax-card { padding: 14px; }
  .tax-root { gap: 5px; }
  .tax-concept { font-size: 30px; }
  .tax-child { font-size: 13px; }
  .sub-children .tax-child { font-size: 12px; }
  .tax-children { margin-top: 12px; }
  .toolbar { padding: 10px; border-radius: 6px; }
  .toolbar-search { gap: 6px; }
  .sort { max-width: 112px; font-size: 11px; padding: 8px 4px; }
  .search input { font-size: 12px; padding: 8px 4px; }
  .search { padding: 0 6px; }
  .toolbar-filters { gap: 8px; max-height: 24vh; overflow-y: auto; }
  .result-bar { grid-template-columns: 1fr auto; gap: 3px 8px; margin-top: 5px; padding-top: 5px; font-size: 11px; }
  .active-cat { grid-column: 1 / -1; }
  .clear { font-size: 11px; }
  .paper { grid-template-columns: 1fr; padding: 13px 14px; gap: 8px; }
  .paper-code { justify-self: start; }
  .paper h3 { font-size: 14px; }
  .paper-authors { font-size: 11.5px; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  .paper { transition: none; }
}
"""


BODY = """
<a class="skip-link" href="#papers" data-i18n="nav.skip">Skip to papers</a>
<nav class="top-nav" aria-label="Main navigation" data-i18n-aria-label="nav.label">
  <div class="container nav-inner">
    <a class="nav-brand" href="#top"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>KV Cache Optimization</a>
    <div class="nav-links">
      <a href="#taxonomy" data-i18n="nav.taxonomy">Taxonomy</a>
      <a href="#papers" data-i18n="nav.papers">Papers</a>
      <a href="https://aclanthology.org/2026.findings-acl.1916/" target="_blank" rel="noopener" data-i18n="nav.survey">Survey</a>
      <a href="https://github.com/jjiantong/Awesome-KV-Cache-Optimization" target="_blank" rel="noopener">GitHub</a>
    </div>
    <button type="button" class="lang-toggle" id="lang-toggle" title="Switch language"><span id="lang-label">中文</span></button>
  </div>
</nav>
<main id="top">
<header class="hero">
  <div class="container">
    <div class="hero-layout">
      <div>
        <div class="eyebrow"><a href="https://aclanthology.org/2026.findings-acl.1916/" target="_blank" rel="noopener">ACL 2026 Findings</a><span aria-hidden="true">·</span><span class="update-dot" aria-hidden="true"></span><span data-i18n="hero.updated">Continuously updated</span></div>
        <h1>Awesome KV Cache Optimization</h1>
        <p class="tagline" data-i18n="hero.tagline">A system-aware map of KV-centric infrastructure for LLM serving.</p>
        <p class="hero-explainer" data-i18n="hero.explainer">Explore the field through when KV is executed, where it is placed, and how it is represented.</p>
        <div class="hero-actions">
          <a class="button primary" href="#papers" data-i18n="hero.explore">Explore Papers</a>
          <a class="button" href="https://aclanthology.org/2026.findings-acl.1916/" target="_blank" rel="noopener" data-i18n="hero.survey">Read the Survey</a>
          <a class="button" href="https://github.com/jjiantong/Awesome-KV-Cache-Optimization" target="_blank" rel="noopener">GitHub</a>
        </div>
      </div>
      <div class="hero-map">
        <p class="map-label" data-i18n="hero.dimensions">Three behavior dimensions</p>
        <div class="dimension temporal"><span class="concept" data-i18n="tax.concept.temporal">WHEN</span><div><span class="identity" data-i18n="tax.identity.temporal">Temporal</span><span class="meaning" data-i18n="tax.sub.temporal">Execution & Scheduling</span></div></div>
        <div class="dimension spatial"><span class="concept" data-i18n="tax.concept.spatial">WHERE</span><div><span class="identity" data-i18n="tax.identity.spatial">Spatial</span><span class="meaning" data-i18n="tax.sub.spatial">Placement & Migration</span></div></div>
        <div class="dimension structural"><span class="concept" data-i18n="tax.concept.structural">HOW</span><div><span class="identity" data-i18n="tax.identity.structural">Structural</span><span class="meaning" data-i18n="tax.sub.structural">Representation & Retention</span></div></div>
      </div>
    </div>
    <div class="hero-stats" id="hero-stats"></div>
  </div>
</header>

<section id="taxonomy">
  <div class="container">
    <h2 class="section-title" data-i18n="tax.title">Taxonomy at a Glance</h2>
    <p class="section-desc" data-i18n="tax.desc">Explore system-aware KV cache optimization through three behavior dimensions. Select any category to filter the papers below.</p>
    <div class="tax-grid" id="tax-grid"></div>
    <p class="tax-note" data-i18n="tax.note"></p>
  </div>
</section>

<section id="papers">
  <div class="container">
    <h2 class="section-title" data-i18n="papers.title">Paper Explorer</h2>
    <p class="section-desc" id="papers-section-desc"></p>

    <div class="toolbar">
      <div class="toolbar-search">
        <label class="search" for="search">
          <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5"/></svg>
          <span class="sr-only" data-i18n="papers.search">Search papers</span>
          <input id="search" type="search" data-i18n-placeholder="papers.search_placeholder" placeholder="Search title, author, or technique..." autocomplete="off"/>
        </label>
        <select class="sort" id="sort" aria-label="Sort papers" data-i18n-aria-label="sort.label">
          <option value="default" data-i18n="sort.default">Sort: Default</option>
          <option value="newest" data-i18n="sort.newest">Sort: Newest venue</option>
          <option value="alpha" data-i18n="sort.alpha">Sort: A → Z</option>
          <option value="code-first" data-i18n="sort.code_first">Sort: Has code first</option>
        </select>
      </div>
      <details class="filter-options" id="filter-options" open>
        <summary data-i18n="filter.options">Filters</summary>
        <div class="toolbar-filters">
          <div class="filter-group">
            <span class="label" data-i18n="filter.venue">Venue</span>
            <button type="button" class="chip active" data-filter="venue" data-value="all" data-i18n="chip.all" aria-pressed="true">All</button>
            <button type="button" class="chip" data-filter="venue" data-value="systems" data-i18n="chip.systems" aria-pressed="false">Systems</button>
            <button type="button" class="chip" data-filter="venue" data-value="ai-ml" data-i18n="chip.ai_ml" aria-pressed="false">AI/ML</button>
            <button type="button" class="chip" data-filter="venue" data-value="preprint" data-i18n="chip.preprint" aria-pressed="false">Preprint</button>
            <button type="button" class="chip" data-filter="venue" data-value="resource" data-i18n="chip.resource" aria-pressed="false">Resource</button>
          </div>

          <div class="filter-group">
            <span class="label" data-i18n="filter.code">Code</span>
            <button type="button" class="chip active" data-filter="code" data-value="any" data-i18n="chip.any" aria-pressed="true">Any</button>
            <button type="button" class="chip" data-filter="code" data-value="yes" data-i18n="chip.has_code" aria-pressed="false">Has code</button>
            <button type="button" class="chip" data-filter="code" data-value="no" data-i18n="chip.no_code" aria-pressed="false">No code</button>
          </div>

          <div class="filter-group">
            <span class="label" data-i18n="filter.more">More</span>
            <button type="button" class="chip" data-filter="new" data-value="yes" data-i18n="chip.new" aria-pressed="false">🆕 New</button>
            <button type="button" class="chip" data-filter="award" data-value="yes" data-i18n="chip.award" aria-pressed="false">🏆 Award</button>
          </div>
        </div>
      </details>
      <div class="result-bar">
        <span class="active-cat" id="active-cat"></span>
        <span id="result-summary" role="status" aria-live="polite" aria-atomic="true"></span>
        <button type="button" class="clear" id="clear-filters" hidden data-i18n="result.clear">Clear filters</button>
      </div>
    </div>

    <div class="papers" id="papers-list"></div>
  </div>
</section>

</main>
<footer>
  <div class="container">
    <div>
      <h5 data-i18n="footer.citation">Citation</h5>
      <p data-i18n="footer.cite_desc">If you find this resource helpful, please cite our survey:</p>
<pre>@inproceedings{jiang2026towards,
  title     = "Towards Efficient Large Language Model Serving: A Survey on System-Aware {KV} Cache Optimization",
  author    = "Jiang, Jiantong and Yang, Peiyu and Zhang, Rui and Liu, Feng",
  booktitle = "Findings of ACL 2026",
  year      = "2026",
  url       = "https://aclanthology.org/2026.findings-acl.1916/"
}</pre>
    </div>
    <div>
      <h5 data-i18n="footer.links">Links</h5>
      <p>
        <a href="https://aclanthology.org/2026.findings-acl.1916/">📄 ACL Anthology</a><br/>
        <a href="https://github.com/jjiantong/Awesome-KV-Cache-Optimization">⭐ GitHub Repository</a><br/>
        <a href="https://github.com/jjiantong/Awesome-KV-Cache-Optimization/blob/main/README.md" data-i18n="footer.full_readme">📖 Full README</a>
      </p>
      <p style="margin-top: 12px; color: var(--text-mute); font-size: 12px;" data-i18n-html="footer.note">
        Maintained by the ACL 2026 survey authors.<br/>
        Page generated from README.md via <code>build_page.py</code>.
      </p>
    </div>
  </div>
</footer>
"""

SCRIPT = r"""
(function() {
  "use strict";
  const DATA = window.__DATA__;
  const papers = DATA.papers;

  // ---- I18N ----
  const I18N = {
    en: {
      'hero.tagline': 'A system-aware map of KV-centric infrastructure for LLM serving.',
      'stats.total': 'Papers',
      'stats.with_code': 'With Code',
      'stats.systems': 'Systems Venues',
      'stats.ai_ml': 'AI/ML Venues',
      'stats.preprint': 'Preprints',
      'stats.resource': 'Resources',
      'stats.new': 'Newly Added',
      'stats.awarded': 'Awarded',
      'tax.title': 'Taxonomy at a Glance',
      'tax.desc': 'Explore system-aware KV cache optimization through three behavior dimensions. Select any category to filter the papers below.',
      'tax.count': '{n} papers',
      'tax.sub.temporal': 'Execution & Scheduling',
      'tax.sub.spatial': 'Placement & Migration',
      'tax.sub.structural': 'Representation & Retention',
      'papers.title': 'Paper Explorer',
      'papers.desc': 'Search and filter {total} curated papers by keyword, venue, code availability, and category.',
      'papers.search_placeholder': 'Search title, author, or technique...',
      'filter.venue': 'Venue',
      'filter.code': 'Code',
      'filter.more': 'More',
      'chip.all': 'All',
      'chip.any': 'Any',
      'chip.systems': 'Systems',
      'chip.ai_ml': 'AI/ML',
      'chip.preprint': 'Preprint',
      'chip.resource': 'Resource',
      'chip.has_code': 'Has code',
      'chip.no_code': 'No code',
      'chip.new': '🆕 New',
      'chip.award': '🏆 Award',
      'sort.default': 'Sort: Default',
      'sort.newest': 'Sort: Newest venue',
      'sort.alpha': 'Sort: A → Z',
      'sort.code_first': 'Sort: Has code first',
      'result.summary': 'Showing {shown} of {total} papers',
      'result.clear': 'Clear filters',
      'result.empty': 'No papers match the current filters.',
      'result.empty_clear': 'Clear filters',
      'result.cat_label': '{path}',
      'badge.new': '🆕 New',
      'badge.systems_default': 'Systems',
      'badge.ai_ml_default': 'AI/ML',
      'badge.preprint_default': 'Preprint',
      'badge.resource_default': 'Resource',
      'paper.code': 'Code',
      'paper.no_code': '— no code —',
      'paper.also_in': 'Also in',
      'footer.citation': 'Citation',
      'footer.cite_desc': 'If you find this resource helpful, please cite our survey:',
      'footer.links': 'Links',
      'footer.full_readme': '📖 Full README',
      'footer.note': 'Maintained by the ACL 2026 survey authors.<br/>Page generated from README.md via <code>build_page.py</code>.',
      'lang.switch_to': '中文',
      'nav.label': 'Main navigation',
      'nav.skip': 'Skip to papers',
      'nav.taxonomy': 'Taxonomy',
      'nav.papers': 'Papers',
      'nav.survey': 'Survey',
      'hero.updated': 'Continuously updated',
      'hero.explainer': 'Explore the field through when KV is executed, where it is placed, and how it is represented.',
      'hero.explore': 'Explore Papers',
      'hero.survey': 'Read the Survey',
      'hero.dimensions': 'Three behavior dimensions',
      'tax.note': 'Counts include cross-category papers. Each paper is counted once per category and appears in a single card.',
      'tax.identity.temporal': 'Temporal',
      'tax.identity.spatial': 'Spatial',
      'tax.identity.structural': 'Structural',
      'tax.concept.temporal': 'WHEN',
      'tax.concept.spatial': 'WHERE',
      'tax.concept.structural': 'HOW',
      'papers.search': 'Search papers',
      'filter.options': 'Filters',
      'sort.label': 'Sort papers',
      'result.all': 'All papers',
      'paper.mechanism': 'Mechanism',
      'lang.title': '切换为中文',
    },
    zh: {
      'hero.tagline': '从系统视角梳理 LLM 服务中以 KV 缓存为核心的基础设施与优化方法。',
      'stats.total': '论文总数',
      'stats.with_code': '含开源代码',
      'stats.systems': '系统会议',
      'stats.ai_ml': 'AI/ML 会议',
      'stats.preprint': '预印本',
      'stats.resource': '资源',
      'stats.new': '新近收录',
      'stats.awarded': '获奖论文',
      'tax.title': '分类体系总览',
      'tax.desc': '从时间、空间和结构三个行为维度理解系统级 KV 缓存优化。点击任意分类，即可筛选下方论文。',
      'tax.count': '{n} 篇论文',
      'tax.sub.temporal': '执行与调度',
      'tax.sub.spatial': '放置与迁移',
      'tax.sub.structural': '表示与保留',
      'papers.title': '论文浏览',
      'papers.desc': '按关键词、会议、代码与分类搜索和筛选共 {total} 篇论文。',
      'papers.search_placeholder': '搜索标题、作者或技术...',
      'filter.venue': '会议',
      'filter.code': '代码',
      'filter.more': '更多',
      'chip.all': '全部',
      'chip.any': '不限',
      'chip.systems': '系统',
      'chip.ai_ml': 'AI/ML',
      'chip.preprint': '预印本',
      'chip.resource': '资源',
      'chip.has_code': '有代码',
      'chip.no_code': '无代码',
      'chip.new': '🆕 新增',
      'chip.award': '🏆 获奖',
      'sort.default': '排序:默认',
      'sort.newest': '排序:最新会议',
      'sort.alpha': '排序:A → Z',
      'sort.code_first': '排序:有代码优先',
      'result.summary': '显示 {shown} / {total} 篇论文',
      'result.clear': '清除筛选',
      'result.empty': '没有论文匹配当前筛选条件。',
      'result.empty_clear': '清除筛选',
      'result.cat_label': '{path}',
      'badge.new': '🆕 新增',
      'badge.systems_default': '系统',
      'badge.ai_ml_default': 'AI/ML',
      'badge.preprint_default': '预印本',
      'badge.resource_default': '资源',
      'paper.code': '代码',
      'paper.no_code': '— 暂无代码 —',
      'paper.also_in': '同时属于',
      'footer.citation': '引用',
      'footer.cite_desc': '如果本资源对您有帮助,请引用我们的综述:',
      'footer.links': '相关链接',
      'footer.full_readme': '📖 完整 README',
      'footer.note': '由 ACL 2026 综述作者维护。<br/>本页面由 README.md 通过 <code>build_page.py</code> 自动生成。',
      'lang.switch_to': 'EN',
      'nav.label': '主导航',
      'nav.skip': '跳转到论文',
      'nav.taxonomy': '分类体系',
      'nav.papers': '论文',
      'nav.survey': '综述',
      'hero.updated': '持续更新',
      'hero.explainer': '沿着何时执行、何处存放、如何表示三条主线，探索 KV 缓存优化研究。',
      'hero.explore': '浏览论文',
      'hero.survey': '阅读综述',
      'hero.dimensions': '三个行为维度',
      'tax.note': '分类计数包含交叉归类的论文；同一论文在每个分类中仅计数一次，并始终只显示一张卡片。',
      'tax.identity.temporal': '时间维度',
      'tax.identity.spatial': '空间维度',
      'tax.identity.structural': '结构维度',
      'tax.concept.temporal': '何时',
      'tax.concept.spatial': '何处',
      'tax.concept.structural': '如何',
      'papers.search': '搜索论文',
      'filter.options': '筛选条件',
      'sort.label': '论文排序',
      'result.all': '全部论文',
      'paper.mechanism': '机制',
      'lang.title': 'Switch to English',
    }
  };

  // Category name translations (English → Chinese)
  const CAT_ZH = {
    'Temporal — Execution & Scheduling': '时间维度 — 执行与调度',
    'Spatial — Placement & Migration': '空间维度 — 放置与迁移',
    'Structural — Representation & Retention': '结构维度 — 表示与保留',
    'KV-Centric Scheduling': 'KV 感知调度',
    'Pipelining & Overlapping': '流水线与重叠',
    'Hardware-aware Execution': '硬件感知执行',
    'Disaggregated Inference': '解耦推理',
    'Compute Offloading': '计算卸载',
    'Memory Hierarchy KV Orchestration': '内存层级 KV 编排',
    'Cross-device Memory Hierarchy': '跨设备内存层级',
    'Intra-GPU Memory Hierarchy': 'GPU 片内内存层级',
    'Compute Device KV Orchestration': '计算设备 KV 编排',
    'KV Cache Compression': 'KV 缓存压缩',
    'Quantization': '量化',
    'Low-rank Approximation': '低秩近似',
    'Structural Compression': '结构化压缩',
    'Codec-based Compression': '编解码压缩',
    'KV Cache Retention Management': 'KV 缓存保留管理',
    'Allocation & Reuse': '分配与复用',
    'Eviction': '驱逐',
    'Hardware-Specialized Execution': '专用硬件执行',
    'Tools, Simulators & Benchmarking Resources': '工具、模拟器与基准测试资源',
  };

  let lang = detectLang();

  function detectLang() {
    try {
      const saved = localStorage.getItem('kv-cache-lang');
      if (saved === 'en' || saved === 'zh') return saved;
    } catch (e) {}
    const nav = (navigator.language || navigator.userLanguage || 'en').toLowerCase();
    return nav.startsWith('zh') ? 'zh' : 'en';
  }

  function t(key, vars) {
    let s = (I18N[lang] && I18N[lang][key]) || (I18N.en && I18N.en[key]) || key;
    if (vars) {
      Object.keys(vars).forEach(k => {
        s = s.split('{' + k + '}').join(String(vars[k]));
      });
    }
    return s;
  }

  function translateCat(name) {
    if (lang === 'zh' && CAT_ZH[name]) return CAT_ZH[name];
    return name;
  }

  function applyLang() {
    // Static text via data-i18n
    document.querySelectorAll('[data-i18n]').forEach(el => {
      el.textContent = t(el.dataset.i18n);
    });
    // Static HTML via data-i18n-html
    document.querySelectorAll('[data-i18n-html]').forEach(el => {
      el.innerHTML = t(el.dataset.i18nHtml);
    });
    // Placeholders
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      el.placeholder = t(el.dataset.i18nPlaceholder);
    });
    document.querySelectorAll('[data-i18n-aria-label]').forEach(el => {
      el.setAttribute('aria-label', t(el.dataset.i18nAriaLabel));
    });
    document.getElementById('lang-label').textContent = t('lang.switch_to');
    document.getElementById('lang-toggle').title = t('lang.title');
    document.getElementById('lang-toggle').setAttribute('aria-label', t('lang.title'));
    // Section descriptions (with placeholders)
    const descEl = document.getElementById('papers-section-desc');
    if (descEl) {
      descEl.innerHTML = t('papers.desc', { total: '<strong>' + papers.length + '</strong>' });
    }
    // Re-render dynamic parts
    renderHeroStats();
    renderTaxGrid();
    render();
  }

  // ---- HERO STATS ----
  function renderHeroStats() {
    const heroStats = document.getElementById('hero-stats');
    const s = DATA.stats;
    const items = [
      { num: s.total, label: t('stats.total') },
      { num: s.with_code, label: t('stats.with_code') },
      { num: s.systems, label: t('stats.systems') },
      { num: s.new, label: t('stats.new') },
    ];
    heroStats.innerHTML = items.map(it =>
      `<div class="hero-stat"><div class="num">${it.num}</div><div class="label">${escapeHtml(it.label)}</div></div>`
    ).join('');
  }

  // ---- TAXONOMY TREE ----
  function behaviorClass(name) {
    const n = name.toLowerCase();
    if (n.includes('temporal')) return 'temporal';
    if (n.includes('spatial')) return 'spatial';
    if (n.includes('structural')) return 'structural';
    return '';
  }
  function renderTaxNode(path, node) {
    const keys = Object.keys(node._children);
    if (!keys.length) return '';
    return `<div class="${path.length === 1 ? 'tax-children' : 'sub-children'}">` +
      keys.map(k => {
        const child = node._children[k];
        const childPath = [...path, k];
        return `<div class="tax-group">
          <button type="button" class="tax-child" data-cat="${escapeAttr(childPath.join('|'))}" aria-pressed="${state.cat === childPath.join('|')}">
            <span class="name">${escapeHtml(translateCat(k))}</span>
            <span class="count">${child._count}</span>
          </button>
          ${renderTaxNode(childPath, child)}
        </div>`;
      }).join('') + '</div>';
  }

  function renderTaxCard(name, node) {
    const cls = behaviorClass(name) || 'resource';
    const identity = cls === 'resource' ? translateCat(name) : t('tax.identity.' + cls);
    const concept = cls === 'resource' ? '' : `<span class="tax-concept">${escapeHtml(t('tax.concept.' + cls))}</span>`;
    const sub = cls === 'resource' ? '' : `<span class="tax-sub">${escapeHtml(t('tax.sub.' + cls))}</span>`;
    return `<div class="tax-card ${cls}">
      <h3><button type="button" class="tax-root" data-cat="${escapeAttr(name)}" aria-pressed="${state.cat === name}">
        <span class="tax-heading"><span class="tax-identity">${escapeHtml(identity)}</span><span class="tax-count">${escapeHtml(t('tax.count', { n: node._count }))}</span></span>
        ${concept}${sub}
      </button></h3>
      ${renderTaxNode([name], node)}
    </div>`;
  }

  function updateActiveTaxonomy() {
    document.querySelectorAll('[data-cat]').forEach(el => {
      const active = el.dataset.cat === state.cat;
      el.classList.toggle('active', active);
      el.setAttribute('aria-pressed', String(active));
    });
  }

  function renderTaxGrid() {
    document.getElementById('tax-grid').innerHTML = Object.keys(DATA.taxonomy)
      .map(k => renderTaxCard(k, DATA.taxonomy[k])).join('');
    updateActiveTaxonomy();
  }

  // ---- FILTER STATE ----
  const state = {
    search: '',
    venue: 'all',
    code: 'any',
    new: '',
    award: '',
    cat: '',
    sort: 'default',
  };

  function updateActiveChips() {
    document.querySelectorAll('.chip').forEach(c => {
      const f = c.dataset.filter;
      const v = c.dataset.value;
      if (!f) return;
      if (f === 'venue') c.classList.toggle('active', state.venue === v);
      else if (f === 'code') c.classList.toggle('active', state.code === v);
      else if (f === 'new') c.classList.toggle('active', state.new === v);
      else if (f === 'award') c.classList.toggle('active', state.award === v);
      c.setAttribute('aria-pressed', String(c.classList.contains('active')));
    });
  }

  function clearCategory() {
    state.cat = '';
    updateActiveTaxonomy();
  }

  function renderResultBar() {
    const el = document.getElementById('active-cat');
    el.className = 'active-cat ' + (state.cat ? behaviorClass(state.cat.split('|')[0]) || 'resource' : '');
    el.textContent = state.cat
      ? t('result.cat_label', { path: state.cat.split('|').map(translateCat).join(' › ') })
      : t('result.all');
    el.title = el.textContent;
  }

  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const f = chip.dataset.filter;
      const v = chip.dataset.value;
      if (f === 'venue') state.venue = (state.venue === v && v === 'all') ? 'all' : (state.venue === v ? 'all' : v);
      else if (f === 'code') state.code = (state.code === v && v === 'any') ? 'any' : (state.code === v ? 'any' : v);
      else if (f === 'new') state.new = state.new ? '' : v;
      else if (f === 'award') state.award = state.award ? '' : v;
      if (f === 'venue' && v === 'all') state.venue = 'all';
      if (f === 'code' && v === 'any') state.code = 'any';
      updateActiveChips();
      render();
    });
  });

  // Match category prefixes against every placement, at every depth.
  function matchesCategory(paper, category) {
    const parts = category.split('|');
    return [paper.path, ...(paper.also_in || [])].some(path =>
      parts.every((part, i) => path[i] === part)
    );
  }

  document.getElementById('tax-grid').addEventListener('click', (e) => {
    const el = e.target.closest('button[data-cat]');
    if (!el) return;
    state.cat = state.cat === el.dataset.cat ? '' : el.dataset.cat;
    updateActiveTaxonomy();
    render();
  });

  // Search
  const searchInput = document.getElementById('search');
  searchInput.addEventListener('input', () => {
    state.search = searchInput.value.trim().toLowerCase();
    render();
  });

  // Sort
  document.getElementById('sort').addEventListener('change', (e) => {
    state.sort = e.target.value;
    render();
  });

  // Clear filters
  document.getElementById('clear-filters').addEventListener('click', () => {
    state.search = '';
    state.venue = 'all';
    state.code = 'any';
    state.new = '';
    state.award = '';
    state.sort = 'default';
    clearCategory();
    searchInput.value = '';
    document.getElementById('sort').value = 'default';
    updateActiveChips();
    render();
  });

  // Lang toggle
  document.getElementById('lang-toggle').addEventListener('click', () => {
    lang = (lang === 'en') ? 'zh' : 'en';
    try { localStorage.setItem('kv-cache-lang', lang); } catch (e) {}
    if (document.documentElement) {
      document.documentElement.lang = (lang === 'zh') ? 'zh-CN' : 'en';
    }
    applyLang();
  });

  // ---- ESCAPE ----
  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
  function escapeAttr(s) {
    return escapeHtml(s).replace(/'/g, '&#39;');
  }

  // ---- RENDER PAPERS ----
  function matches(p) {
    if (state.venue !== 'all' && p.venue_type !== state.venue) return false;
    if (state.code === 'yes' && !p.has_code) return false;
    if (state.code === 'no' && p.has_code) return false;
    if (state.new && !p.is_new) return false;
    if (state.award && !p.award) return false;
    if (state.cat && !matchesCategory(p, state.cat)) return false;
    if (state.search) {
      const hay = (p.title + ' ' + p.authors + ' ' + p.comment + ' ' + (p.venue || '')).toLowerCase();
      if (!hay.includes(state.search)) return false;
    }
    return true;
  }

  function sortPapers(arr) {
    const a = arr.slice();
    if (state.sort === 'alpha') {
      a.sort((x, y) => x.title.localeCompare(y.title));
    } else if (state.sort === 'newest') {
      a.sort((x, y) => (y.year || 0) - (x.year || 0) || x.title.localeCompare(y.title));
    } else if (state.sort === 'code-first') {
      a.sort((x, y) => (y.has_code ? 1 : 0) - (x.has_code ? 1 : 0) || x.title.localeCompare(y.title));
    }
    return a;
  }

  function venueBadge(p) {
    const v = p.venue || t('badge.' + p.venue_type + '_default');
    if (p.venue_type === 'systems') {
      return `<span class="badge systems">⚙️ ${escapeHtml(v)}</span>`;
    } else if (p.venue_type === 'ai-ml') {
      return `<span class="badge ai-ml">🎓 ${escapeHtml(v)}</span>`;
    } else if (p.venue_type === 'resource') {
      return `<span class="badge resource">🛠️ ${escapeHtml(v)}</span>`;
    } else {
      return `<span class="badge preprint">📄 ${escapeHtml(v)}</span>`;
    }
  }

  function renderPaper(p) {
    const badges = [];
    if (p.is_new) badges.push(`<span class="badge new">${t('badge.new')}</span>`);
    badges.push(venueBadge(p));
    if (p.award) badges.push(`<span class="badge award">🏆 ${escapeHtml(p.award)}</span>`);

    const codeHtml = p.has_code && p.github_url
      ? `<a href="${escapeAttr(p.github_url)}" target="_blank" rel="noopener" class="code-button"><span class="code-icon" aria-hidden="true">&lt;&gt;</span>${escapeHtml(t('paper.code'))}</a>`
      : `<span class="no-code">${escapeHtml(t('paper.no_code'))}</span>`;

    function renderPath(path, secondary) {
      const cls = behaviorClass(path[0]) || 'resource';
      const joined = path.join('|');
      const matched = state.cat && (joined === state.cat || joined.startsWith(state.cat + '|'));
      const crumbs = path.map((name, i) => `<span${i === 0 ? ' class="path-root"' : ''}>${escapeHtml(translateCat(name))}</span>`)
        .join('<span class="sep" aria-hidden="true">›</span>');
      return `<div class="taxonomy-path ${cls}${secondary ? ' secondary-path' : ''}${matched ? ' matched' : ''}">${secondary ? `<span class="also">${escapeHtml(t('paper.also_in'))}:</span>` : ''}${crumbs}</div>`;
    }
    const pathHtml = renderPath(p.path, false) + (p.also_in || []).map(path => renderPath(path, true)).join('');

    const titleHtml = p.url
      ? `<h3><a href="${escapeAttr(p.url)}" target="_blank" rel="noopener">${escapeHtml(p.title)}</a></h3>`
      : `<h3>${escapeHtml(p.title)}</h3>`;

    return `<article class="paper">
      <div class="paper-main">
        <div class="paper-meta">${badges.join('')}</div>
        ${titleHtml}
        ${p.authors ? `<div class="paper-authors">${escapeHtml(p.authors)}</div>` : ''}
        ${p.comment ? `<div class="paper-comment"><span class="mechanism-label">${escapeHtml(t('paper.mechanism'))} ·</span><span class="mechanism-text">${escapeHtml(p.comment)}</span></div>` : ''}
        <div class="paper-path">${pathHtml}</div>
      </div>
      <div class="paper-code">${codeHtml}</div>
    </article>`;
  }

  function render() {
    const filtered = sortPapers(papers.filter(matches));
    const list = document.getElementById('papers-list');
    if (filtered.length === 0) {
      const emptyClear = `<button type="button" class="clear" id="empty-clear">${escapeHtml(t('result.empty_clear'))}</button>`;
      list.innerHTML = `<div class="empty">${escapeHtml(t('result.empty'))} ${emptyClear}</div>`;
      const ec = document.getElementById('empty-clear');
      if (ec) ec.addEventListener('click', () => document.getElementById('clear-filters').click());
    } else {
      list.innerHTML = filtered.map(renderPaper).join('');
    }
    document.getElementById('result-summary').innerHTML = t('result.summary', {
      shown: '<strong>' + filtered.length + '</strong>',
      total: '<strong>' + papers.length + '</strong>',
    });
    document.getElementById('clear-filters').hidden =
      !(state.search || state.venue !== 'all' || state.code !== 'any' || state.new || state.award || state.cat);
    renderResultBar();
  }

  // Keep the sticky toolbar below the actual navbar, including wrapped mobile text.
  const topNav = document.querySelector('.top-nav');
  function updateNavHeight() {
    document.documentElement.style.setProperty('--nav-height', topNav.getBoundingClientRect().height + 'px');
  }
  new ResizeObserver(updateNavHeight).observe(topNav);
  updateNavHeight();

  const mobileLayout = window.matchMedia('(max-width: 720px)');
  const filterOptions = document.getElementById('filter-options');
  function updateFilterDisclosure() { filterOptions.open = !mobileLayout.matches; }
  mobileLayout.addEventListener('change', updateFilterDisclosure);
  updateFilterDisclosure();

  // ---- BOOTSTRAP ----
  if (document.documentElement) {
    document.documentElement.lang = (lang === 'zh') ? 'zh-CN' : 'en';
  }
  updateActiveChips();
  applyLang();
})();
"""


if __name__ == "__main__":
    main()
