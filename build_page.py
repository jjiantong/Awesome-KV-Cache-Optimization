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
  /* Shared visual vocabulary: type, spacing, surfaces and taxonomy identity. */
  --space-1: 4px; --space-2: 8px; --space-3: 12px; --space-4: 16px;
  --space-6: 24px; --space-8: 32px; --space-12: 48px; --space-16: 64px;
  --font-xs: 11px; --font-meta: 12px; --font-sm: 13px; --font-body: 15px;
  --font-paper: 16px; --font-subtitle: 20px; --font-section: 24px; --font-concept: 32px;
  --font-hero: clamp(32px, 3.5vw, 46px);
  --radius-taxonomy: 8px; --radius-paper: 4px; --radius-control: 4px; --radius-tag: 2px;
  --bg: #fff; --bg-soft: #fafbfd; --bg-utility: rgba(255,255,255,.97);
  --border: #dfe4ea; --border-soft: #edf0f3;
  --text: #202c3c; --text-soft: #526174; --text-mute: #637286;
  --accent: #273d57; --link: #1d4ed8;
  --temporal: #1d4ed8; --temporal-bg: #eff6ff; --temporal-border: #bfdbfe;
  --spatial: #047857; --spatial-bg: #ecfdf5; --spatial-border: #a7f3d0;
  --structural: #7e22ce; --structural-bg: #faf5ff; --structural-border: #e9d5ff;
  --nav-height: 65px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: calc(var(--nav-height) + var(--space-6)); }
body {
  margin: 0; color: var(--text); background: var(--bg);
  font: var(--font-body)/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
button, input, select { font: inherit; }
button { cursor: pointer; }
button, a, input, select, summary { -webkit-tap-highlight-color: transparent; }
:focus-visible { outline: 2px solid var(--link); outline-offset: var(--space-1); }
[hidden] { display: none !important; }
.container { max-width: 1200px; margin: 0 auto; padding: 0 var(--space-8); }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
.skip-link { position: fixed; top: -60px; left: var(--space-4); z-index: 100; background: white; padding: var(--space-2) var(--space-4); }
.skip-link:focus { top: var(--space-2); }
.temporal { --tax-color: var(--temporal); --tax-bg: var(--temporal-bg); --tax-border: var(--temporal-border); }
.spatial { --tax-color: var(--spatial); --tax-bg: var(--spatial-bg); --tax-border: var(--spatial-border); }
.structural { --tax-color: var(--structural); --tax-bg: var(--structural-bg); --tax-border: var(--structural-border); }
.resource { --tax-color: var(--text-soft); --tax-bg: var(--bg-soft); --tax-border: var(--border); }
/* Navigation */
.top-nav { position: sticky; top: 0; z-index: 30; border-bottom: 1px solid var(--border-soft); background: var(--bg-utility); backdrop-filter: blur(12px); }
.nav-inner { min-height: 64px; display: grid; grid-template-columns: 1fr auto auto; gap: var(--space-6); align-items: center; }
.nav-brand { color: var(--text); font-size: 14px; font-weight: 650; letter-spacing: -.02em; }
.nav-brand:hover { text-decoration: none; }
.brand-mark { display: inline-flex; gap: 3px; margin-right: var(--space-2); vertical-align: middle; }
.brand-mark i { width: 3px; height: 16px; border-radius: 1px; background: var(--temporal); }
.brand-mark i:nth-child(2) { background: var(--spatial); }
.brand-mark i:nth-child(3) { background: var(--structural); }
.nav-links { display: flex; gap: var(--space-6); align-items: center; }
.nav-links a { color: var(--text-soft); font-size: var(--font-sm); padding: var(--space-2) 0; }
.nav-links a:hover { color: var(--text); }
.lang-toggle { padding: var(--space-1) var(--space-2); border: 1px solid var(--border); border-radius: var(--radius-control); background: transparent; color: var(--text-soft); font-size: var(--font-meta); }
.lang-toggle:hover { color: var(--text); border-color: var(--text-mute); }
/* Hero: a research map, with an unboxed statistical footnote. */
.hero { padding: var(--space-12) 0; border-bottom: 1px solid var(--border-soft); }
.hero-layout { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(380px, 1fr); gap: var(--space-12); align-items: center; }
.hero-copy { min-width: 0; }
.eyebrow { display: inline-flex; align-items: center; flex-wrap: wrap; gap: var(--space-2); font-size: var(--font-xs); letter-spacing: .04em; color: var(--text-soft); margin-bottom: var(--space-4); }
.update-dot { width: 5px; height: 5px; background: var(--spatial); border-radius: 50%; }
.hero h1 { font-size: var(--font-hero); line-height: 1.14; letter-spacing: -.035em; margin: 0 0 var(--space-6); font-weight: 700; max-width: 620px; }
.hero .tagline { font-size: var(--font-subtitle); line-height: 1.5; margin: 0 0 var(--space-3); max-width: 44ch; }
.hero-explainer { color: var(--text-soft); font-size: var(--font-sm); max-width: 58ch; margin: 0 0 var(--space-6); }
.hero-actions { display: flex; flex-wrap: wrap; gap: var(--space-2); }
.button { display: inline-flex; justify-content: center; align-items: center; gap: var(--space-2); border: 1px solid var(--border); border-radius: var(--radius-control); padding: var(--space-2) var(--space-4); background: transparent; color: var(--text); font-size: var(--font-sm); font-weight: 600; }
.button:hover { text-decoration: none; border-color: var(--text-mute); }
.button.primary { background: var(--accent); border-color: var(--accent); color: white; }
.hero-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); margin-top: var(--space-6); padding-top: var(--space-4); border-top: 1px solid var(--border-soft); }
.hero-stat { padding: 0 var(--space-3); border-left: 1px solid var(--border-soft); }
.hero-stat:first-child { border-left: 0; padding-left: 0; }
.hero-stat .num { font-size: 22px; font-weight: 600; line-height: 1.3; font-variant-numeric: tabular-nums; }
.hero-stat .label { font-size: var(--font-xs); color: var(--text-soft); margin-top: var(--space-1); }
/* SVG labels and paths inherit exactly the same tokens as cards and breadcrumbs. */
.hero-map { margin: 0; min-width: 0; }
.taxonomy-visual { display: block; width: 100%; max-width: 480px; height: auto; margin: 0 auto; overflow: visible; }
.visual-arm { fill: none; stroke: var(--tax-color); stroke-width: 1.5; }
.visual-orbit { fill: none; stroke: var(--tax-border); stroke-width: 1.5; }
.visual-point { fill: var(--tax-color); }
.visual-concept { fill: var(--tax-color); font-size: 32px; font-weight: 650; letter-spacing: -.025em; }
.visual-identity { fill: var(--tax-color); font-size: 17px; font-weight: 500; }
.visual-meaning { fill: var(--text-soft); font-size: 16px; }
.visual-core { fill: var(--bg); stroke: var(--border); stroke-width: 1; }
.visual-name { fill: var(--text); font-size: 40px; font-weight: 650; letter-spacing: -.04em; }
.visual-cache { fill: var(--text-soft); font-size: 14px; }
.visual-scope { margin-top: var(--space-3); color: var(--text-mute); font-size: var(--font-xs); letter-spacing: .03em; text-align: center; }
.hero-map-compact { display: none; }
/* The three taxonomy cards are the page's principal framed elements. */
section { padding: var(--space-12) 0; }
section + section { border-top: 1px solid var(--border-soft); }
.section-title { margin: 0 0 var(--space-2); font-size: var(--font-section); font-weight: 650; line-height: 1.3; letter-spacing: -.025em; }
.section-desc { color: var(--text-soft); font-size: var(--font-sm); margin: 0 0 var(--space-8); max-width: 68ch; }
.tax-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--space-6); align-items: start; }
.tax-card { border: 1px solid var(--border); border-top: 2px solid var(--tax-color); border-radius: var(--radius-taxonomy); padding: var(--space-4); min-width: 0; }
.tax-card h3 { margin: 0; }
.tax-root { display: flex; flex-direction: column; align-items: stretch; gap: var(--space-2); width: 100%; padding: var(--space-2); color: var(--tax-color); background: transparent; border: 1px solid transparent; border-radius: var(--radius-control); text-align: left; }
.tax-heading { display: flex; justify-content: space-between; align-items: center; gap: var(--space-2); }
.tax-identity { font-size: var(--font-xs); font-weight: 600; letter-spacing: .12em; text-transform: uppercase; }
.tax-concept { font-size: var(--font-concept); line-height: 1.1; font-weight: 650; letter-spacing: -.03em; }
.tax-sub { font-size: var(--font-sm); color: var(--text-soft); font-weight: 400; }
.tax-count { font-size: var(--font-xs); font-weight: 500; white-space: nowrap; }
.tax-root:hover, .tax-child:hover { background: var(--tax-bg); }
.tax-root.active, .tax-child.active { color: var(--tax-color); background: var(--tax-bg); border-color: var(--tax-border); }
.tax-root:focus-visible, .tax-child:focus-visible { outline-color: var(--tax-color); }
.tax-children { margin-top: var(--space-4); border-top: 1px solid var(--border-soft); padding-top: var(--space-2); }
.tax-child { display: flex; justify-content: space-between; align-items: center; gap: var(--space-2); width: 100%; border: 1px solid transparent; background: transparent; color: var(--text); border-radius: var(--radius-control); padding: var(--space-2); text-align: left; font-size: var(--font-sm); line-height: 1.45; font-weight: 500; }
.tax-child .name { overflow-wrap: anywhere; }
.tax-child .count { color: var(--tax-color); min-width: 24px; padding: 0 var(--space-1); font-size: var(--font-xs); text-align: right; font-variant-numeric: tabular-nums; }
.tax-child.active .name { font-weight: 600; }
.sub-children { margin: 0 0 var(--space-2) var(--space-2); padding-left: var(--space-2); border-left: 1px solid var(--tax-border); }
.sub-children .tax-child { font-size: var(--font-meta); font-weight: 400; color: var(--text-soft); padding: var(--space-1) var(--space-2); }
.sub-children .tax-child.active { color: var(--tax-color); }
.tax-card.resource { grid-column: 1 / -1; padding: var(--space-1) 0; border: 0; border-top: 1px solid var(--border-soft); border-radius: 0; }
.resource .tax-root { flex-direction: row; align-items: center; justify-content: space-between; color: var(--text-soft); }
.resource .tax-heading { width: 100%; }
.resource .tax-identity { font-size: var(--font-meta); font-weight: 400; letter-spacing: 0; text-transform: none; }
.tax-note { margin: var(--space-3) 0 0; font-size: var(--font-meta); color: var(--text-mute); max-width: 80ch; }
/* A flat, sticky utility bar over a quiet bibliography surface. */
#papers { background: var(--bg-soft); }
.toolbar { position: sticky; top: var(--nav-height); z-index: 20; background: var(--bg-utility); backdrop-filter: blur(12px); border-block: 1px solid var(--border); padding: var(--space-3) 0; margin-bottom: var(--space-4); }
.toolbar-search { display: flex; gap: var(--space-3); align-items: center; }
.search { display: flex; align-items: center; flex: 1; min-width: 0; padding: 0 var(--space-2); background: white; border: 1px solid var(--border); border-radius: var(--radius-control); }
.search:focus-within { border-color: var(--link); outline: 2px solid var(--link); outline-offset: 2px; }
.search svg { width: 16px; height: 16px; color: var(--text-mute); flex: none; }
.search input { width: 100%; min-width: 0; border: 0; outline: 0; background: transparent; padding: var(--space-2); font-size: var(--font-sm); color: var(--text); }
.sort { max-width: 180px; padding: var(--space-2); background: white; border: 1px solid var(--border); border-radius: var(--radius-control); color: var(--text-soft); font-size: var(--font-meta); }
.filter-options { margin-top: var(--space-2); }
.filter-options summary { color: var(--text-soft); font-size: var(--font-meta); cursor: pointer; width: fit-content; padding: 2px 0; }
.toolbar-filters { display: flex; flex-wrap: wrap; gap: var(--space-2) var(--space-6); padding: var(--space-1) 0; }
.filter-group { display: flex; flex-wrap: wrap; gap: var(--space-1); align-items: center; }
.filter-group .label { font-size: var(--font-xs); color: var(--text-soft); margin-right: var(--space-1); }
.chip { padding: var(--space-1) var(--space-2); color: var(--text-soft); background: transparent; border: 1px solid var(--border); border-radius: var(--radius-control); font-size: var(--font-xs); white-space: nowrap; }
.chip:hover { border-color: var(--text-mute); }
.chip.active { background: var(--accent); border-color: var(--accent); color: white; }
.result-bar { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: var(--space-2) var(--space-4); align-items: center; border-top: 1px solid var(--border-soft); margin-top: var(--space-2); padding-top: var(--space-2); font-size: var(--font-meta); color: var(--text-soft); }
.active-cat { color: var(--tax-color, var(--text-soft)); min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.clear { border: 0; background: transparent; color: var(--link); font-size: var(--font-meta); padding: var(--space-1) 0; cursor: pointer; }
.clear:hover { text-decoration: underline; }
.papers { display: flex; flex-direction: column; gap: var(--space-2); }
.paper { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: var(--space-4); padding: var(--space-4) var(--space-6); border: 1px solid var(--border-soft); border-radius: var(--radius-paper); background: white; transition: border-color .15s, background .15s; }
.paper:hover { border-color: var(--border); background: #fdfdfe; }
.paper-main { min-width: 0; }
.paper-meta { display: flex; flex-wrap: wrap; gap: var(--space-1); margin-bottom: var(--space-2); }
.badge { display: inline-flex; align-items: center; font-size: var(--font-xs); font-weight: 500; line-height: 1.6; padding: 0 var(--space-1); border: 1px solid transparent; border-radius: var(--radius-tag); }
.badge.new { background: #fdf2f8; color: #be185d; border-color: #fbcfe8; }
.badge.systems { background: #ecfeff; color: #0e7490; border-color: #a5f3fc; }
.badge.ai-ml { background: var(--temporal-bg); color: var(--temporal); border-color: var(--temporal-border); }
.badge.preprint { background: #f3f4f6; color: #4b5563; border-color: #d1d5db; }
.badge.resource { background: #eef2ff; color: #4338ca; border-color: #c7d2fe; }
.badge.award { background: #fffbeb; color: #b45309; border-color: #fde68a; }
.paper h3 { font-size: var(--font-paper); font-weight: 600; line-height: 1.5; margin: 0 0 var(--space-1); overflow-wrap: anywhere; max-width: 90ch; }
.paper h3 a { color: var(--text); }
.paper h3 a:hover { color: var(--link); }
.paper-authors { font-size: var(--font-sm); color: var(--text-soft); margin-bottom: var(--space-2); overflow-wrap: anywhere; }
.paper-comment { font-size: var(--font-sm); color: var(--text-soft); margin-top: var(--space-2); line-height: 1.6; overflow-wrap: anywhere; }
.mechanism-label { font-size: var(--font-xs); font-weight: 500; color: var(--text-mute); margin-right: var(--space-1); }
.paper-path { font-size: var(--font-meta); color: var(--text-soft); margin-top: var(--space-2); }
.taxonomy-path { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--space-1); line-height: 1.6; }
.path-root { color: var(--tax-color); font-weight: 500; }
.taxonomy-path.matched .path-root { background: var(--tax-bg); border-radius: var(--radius-tag); }
.paper-path .sep { color: var(--text-mute); }
.paper-path .also { color: var(--text-mute); font-size: var(--font-xs); }
.secondary-path { margin-top: var(--space-1); }
.paper-code { align-self: start; }
.code-button { display: inline-flex; gap: var(--space-1); align-items: center; padding: var(--space-1) var(--space-2); border: 1px solid var(--border); border-radius: var(--radius-control); color: var(--text-soft); font-size: var(--font-xs); white-space: nowrap; }
.code-button:hover { text-decoration: none; color: var(--text); border-color: var(--text-mute); }
.code-icon { font: var(--font-meta) ui-monospace, SFMono-Regular, monospace; }
.no-code { color: var(--text-mute); font-size: var(--font-xs); }
.empty { padding: var(--space-8) 0; text-align: center; color: var(--text-soft); }
/* An academic closing section, with a typographic BibTeX block. */
footer { border-top: 1px solid var(--border-soft); padding: var(--space-12) 0; font-size: var(--font-sm); color: var(--text-soft); }
.footer-layout { display: grid; grid-template-columns: minmax(0, 2fr) minmax(220px, 1fr); gap: var(--space-12); }
.footer-layout > div { min-width: 0; }
footer h2 { font-size: var(--font-subtitle); font-weight: 600; line-height: 1.3; margin: 0 0 var(--space-2); color: var(--text); letter-spacing: -.02em; }
footer h3 { font-size: var(--font-sm); font-weight: 600; margin: 0 0 var(--space-3); color: var(--text); }
footer p { margin: 0 0 var(--space-4); max-width: 64ch; }
footer pre { margin: var(--space-4) 0 0; padding: var(--space-4); background: var(--bg-soft); border: 0; border-left: 2px solid var(--border); font: var(--font-meta)/1.7 ui-monospace, SFMono-Regular, monospace; max-width: 100%; overflow-x: auto; }
.footer-note { margin-top: var(--space-6); color: var(--text-mute); font-size: var(--font-xs); }
@media (max-width: 1024px) {
  .hero-layout { gap: var(--space-8); }
  .tax-grid { gap: var(--space-3); }
  .tax-card { padding: var(--space-3); }
  .tax-child { font-size: var(--font-meta); }
}
@media (max-width: 768px) {
  .hero-layout { grid-template-columns: 1fr; gap: var(--space-6); }
  .hero-copy { max-width: 620px; }
  .taxonomy-visual { display: none; }
  .hero-map { max-width: 520px; }
  .hero-map-compact { display: grid; grid-template-columns: 64px minmax(0, 1fr); gap: var(--space-6); align-items: center; }
  .compact-core { text-align: center; }
  .compact-name { font-size: 28px; font-weight: 650; letter-spacing: -.04em; line-height: 1.2; }
  .compact-cache { color: var(--text-soft); font-size: var(--font-xs); }
  .compact-branches { border-left: 1px solid var(--border); }
  .dimension { display: grid; grid-template-columns: 64px minmax(0, 1fr); align-items: baseline; gap: var(--space-3); position: relative; padding: var(--space-1) 0 var(--space-1) var(--space-4); }
  .dimension + .dimension { margin-top: var(--space-2); }
  .dimension::before { content: ""; position: absolute; left: -1px; top: 14px; width: 8px; height: 1px; background: var(--tax-color); }
  .dimension .concept { color: var(--tax-color); font-size: 14px; font-weight: 650; }
  .dimension .identity { color: var(--tax-color); font-size: var(--font-meta); font-weight: 500; }
  .dimension .meaning { display: block; color: var(--text-soft); font-size: var(--font-xs); }
  .visual-scope { margin-top: var(--space-3); text-align: left; }
  .tax-grid { grid-template-columns: 1fr; gap: var(--space-4); }
  .tax-card { padding: var(--space-4); }
  .tax-child { font-size: var(--font-sm); }
  .footer-layout { grid-template-columns: 1fr; gap: var(--space-8); }
}
@media (max-width: 720px) {
  .container { padding: 0 var(--space-4); }
  .nav-inner { grid-template-columns: 1fr auto; gap: 0 var(--space-3); padding-top: var(--space-2); padding-bottom: var(--space-1); }
  .nav-brand { font-size: var(--font-sm); }
  .nav-links { grid-row: 2; grid-column: 1 / -1; justify-content: space-between; gap: var(--space-3); }
  .lang-toggle { grid-column: 2; grid-row: 1; }
  .nav-links a { font-size: var(--font-meta); }
  .hero, section, footer { padding: var(--space-8) 0; }
  .hero h1 { margin-bottom: var(--space-4); }
  .hero .tagline { font-size: 17px; }
  .hero-actions .button { padding: var(--space-2) var(--space-3); font-size: var(--font-meta); }
  .hero-stats { margin-top: var(--space-4); }
  .hero-stat { padding: 0 var(--space-2); }
  .hero-stat .num { font-size: 20px; }
  .hero-stat .label { font-size: 10px; }
  .section-desc { margin-bottom: var(--space-6); }
  .toolbar { padding: var(--space-2) 0; }
  .toolbar-search { gap: var(--space-2); }
  .sort { max-width: 112px; font-size: var(--font-xs); padding: var(--space-2) var(--space-1); }
  .search input { font-size: var(--font-meta); padding: var(--space-2) var(--space-1); }
  .search { padding: 0 var(--space-1); }
  .toolbar-filters { gap: var(--space-2); max-height: 24vh; overflow-y: auto; }
  .result-bar { grid-template-columns: 1fr auto; gap: var(--space-1) var(--space-2); margin-top: var(--space-1); padding-top: var(--space-1); font-size: var(--font-xs); }
  .active-cat { grid-column: 1 / -1; }
  .clear { font-size: var(--font-xs); }
  .paper { grid-template-columns: 1fr; padding: var(--space-4); gap: var(--space-2); }
  .paper-code { justify-self: start; }
  .paper h3 { font-size: var(--font-body); }
  .paper-authors, .paper-comment { font-size: var(--font-meta); }
  .paper-path { font-size: var(--font-xs); }
}
@media (max-width: 360px) {
  .hero-map-compact { gap: var(--space-4); grid-template-columns: 52px minmax(0, 1fr); }
  .dimension { grid-template-columns: 56px minmax(0, 1fr); gap: var(--space-2); padding-left: var(--space-3); }
  .hero-stat { padding: 0 var(--space-1); }
  .hero-stat .label { overflow-wrap: anywhere; }
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
      <div class="hero-copy">
        <div class="eyebrow"><a href="https://aclanthology.org/2026.findings-acl.1916/" target="_blank" rel="noopener">ACL 2026 Findings</a><span aria-hidden="true">·</span><span class="update-dot" aria-hidden="true"></span><span data-i18n="hero.updated">Continuously updated</span></div>
        <h1>Awesome KV Cache Optimization</h1>
        <p class="tagline" data-i18n="hero.tagline">A system-aware map of KV-centric infrastructure for LLM serving.</p>
        <p class="hero-explainer" data-i18n="hero.explainer">Explore the field through when KV is executed, where it is placed, and how it is represented.</p>
        <div class="hero-actions">
          <a class="button primary" href="#papers" data-i18n="hero.explore">Explore Papers</a>
          <a class="button" href="https://aclanthology.org/2026.findings-acl.1916/" target="_blank" rel="noopener" data-i18n="hero.survey">Read the Survey</a>
          <a class="button" href="https://github.com/jjiantong/Awesome-KV-Cache-Optimization" target="_blank" rel="noopener">GitHub</a>
        </div>
        <div class="hero-stats" id="hero-stats"></div>
      </div>
      <figure class="hero-map">
        <svg class="taxonomy-visual" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 376" role="img" aria-labelledby="taxonomy-visual-title taxonomy-visual-desc" focusable="false">
          <title id="taxonomy-visual-title" data-i18n="hero.visual_title">sKis: three behavior dimensions of KV cache optimization</title>
          <desc id="taxonomy-visual-desc" data-i18n="hero.visual_desc">KV Cache at the center connects to WHEN, Temporal execution and scheduling; WHERE, Spatial placement and migration; and HOW, Structural representation and retention.</desc>
          <g class="temporal" aria-hidden="true"><path class="visual-orbit" d="M224 124 A84 84 0 0 1 336 124"/><path class="visual-arm" d="M280 106 V128"/><circle class="visual-point" cx="280" cy="106" r="3"/></g>
          <g class="spatial" aria-hidden="true"><path class="visual-orbit" d="M198 174 A84 84 0 0 0 262 270"/><path class="visual-arm" d="M230 222 L166 266"/><circle class="visual-point" cx="166" cy="266" r="3"/></g>
          <g class="structural" aria-hidden="true"><path class="visual-orbit" d="M298 270 A84 84 0 0 0 362 174"/><path class="visual-arm" d="M330 222 L394 266"/><circle class="visual-point" cx="394" cy="266" r="3"/></g>
          <circle class="visual-core" cx="280" cy="188" r="60"/>
          <text class="visual-name" x="280" y="187" text-anchor="middle">sKis</text>
          <text class="visual-cache" x="280" y="212" text-anchor="middle" data-i18n="hero.cache">KV Cache</text>
          <g class="map-node temporal" text-anchor="middle">
            <text class="visual-concept" x="280" y="48" data-i18n="tax.concept.temporal">WHEN</text>
            <text class="visual-identity" x="280" y="74" data-i18n="tax.identity.temporal">Temporal</text>
            <text class="visual-meaning" x="280" y="98" data-i18n="tax.sub.temporal">Execution &amp; Scheduling</text>
          </g>
          <g class="map-node spatial" text-anchor="middle">
            <text class="visual-concept" x="132" y="300" data-i18n="tax.concept.spatial">WHERE</text>
            <text class="visual-identity" x="132" y="326" data-i18n="tax.identity.spatial">Spatial</text>
            <text class="visual-meaning" x="132" y="350" data-i18n="tax.sub.spatial">Placement &amp; Migration</text>
          </g>
          <g class="map-node structural" text-anchor="middle">
            <text class="visual-concept" x="428" y="300" data-i18n="tax.concept.structural">HOW</text>
            <text class="visual-identity" x="428" y="326" data-i18n="tax.identity.structural">Structural</text>
            <text class="visual-meaning" x="428" y="350" data-i18n="tax.sub.structural">Representation &amp; Retention</text>
          </g>
        </svg>
        <div class="hero-map-compact">
          <div class="compact-core"><div class="compact-name">sKis</div><div class="compact-cache" data-i18n="hero.cache">KV Cache</div></div>
          <div class="compact-branches">
            <div class="dimension temporal"><span class="concept" data-i18n="tax.concept.temporal">WHEN</span><div><span class="identity" data-i18n="tax.identity.temporal">Temporal</span><span class="meaning" data-i18n="tax.sub.temporal">Execution &amp; Scheduling</span></div></div>
            <div class="dimension spatial"><span class="concept" data-i18n="tax.concept.spatial">WHERE</span><div><span class="identity" data-i18n="tax.identity.spatial">Spatial</span><span class="meaning" data-i18n="tax.sub.spatial">Placement &amp; Migration</span></div></div>
            <div class="dimension structural"><span class="concept" data-i18n="tax.concept.structural">HOW</span><div><span class="identity" data-i18n="tax.identity.structural">Structural</span><span class="meaning" data-i18n="tax.sub.structural">Representation &amp; Retention</span></div></div>
          </div>
        </div>
        <figcaption class="visual-scope" data-i18n="hero.scope">System-aware · Serving-time · KV-centric</figcaption>
      </figure>
    </div>
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
  <div class="container footer-layout">
    <div>
      <h2 data-i18n="footer.citation">Cite this survey</h2>
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
      <h3 data-i18n="footer.links">Links</h3>
      <p>
        <a href="https://aclanthology.org/2026.findings-acl.1916/">ACL Anthology</a><br/>
        <a href="https://github.com/jjiantong/Awesome-KV-Cache-Optimization" data-i18n="footer.repository">GitHub Repository</a><br/>
        <a href="https://github.com/jjiantong/Awesome-KV-Cache-Optimization/blob/main/README.md" data-i18n="footer.full_readme">Full README</a>
      </p>
      <p class="footer-note" data-i18n-html="footer.note">
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
      'footer.citation': 'Cite this survey',
      'footer.cite_desc': 'If you find this resource helpful, please cite our survey:',
      'footer.links': 'Links',
      'footer.full_readme': 'Full README',
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
      'hero.visual_title': 'sKis: three behavior dimensions of KV cache optimization',
      'hero.visual_desc': 'KV Cache at the center connects to WHEN, Temporal execution and scheduling; WHERE, Spatial placement and migration; and HOW, Structural representation and retention.',
      'hero.cache': 'KV Cache',
      'hero.scope': 'System-aware · Serving-time · KV-centric',
      'footer.repository': 'GitHub Repository',
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
      'footer.citation': '引用本综述',
      'footer.cite_desc': '如果本资源对您有帮助,请引用我们的综述:',
      'footer.links': '相关链接',
      'footer.full_readme': '完整 README',
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
      'hero.visual_title': 'sKis：KV 缓存优化的三个行为维度',
      'hero.visual_desc': '中心的 KV 缓存连接三个维度：何时，时间维度的执行与调度；何处，空间维度的放置与迁移；如何，结构维度的表示与保留。',
      'hero.cache': 'KV 缓存',
      'hero.scope': '系统视角 · 服务阶段 · 以 KV 为核心',
      'footer.repository': 'GitHub 仓库',
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
