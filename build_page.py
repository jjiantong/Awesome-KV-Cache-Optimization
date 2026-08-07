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

    for line in text.split("\n"):
        if line.startswith("## "):
            title = line[3:].strip()
            if any(x in title for x in NON_PAPER_SECTIONS):
                current_path = []
            else:
                current_path = [(2, title)]
        elif line.startswith("### ") and current_path:
            title = line[4:].strip()
            current_path = current_path[:1] + [(3, title)]
        elif line.startswith("#### ") and current_path:
            title = line[5:].strip()
            current_path = current_path[:2] + [(4, title)]
        elif line.startswith("|") and current_path:
            paper = parse_row(line)
            if paper:
                paper["path"] = [name for _, name in current_path]
                papers.append(paper)

    return papers


def build_taxonomy(papers):
    """Build the category tree with paper counts."""
    tree = {}
    for p in papers:
        path = p["path"]
        node = tree
        for i, name in enumerate(path):
            key = name
            if key not in node:
                node[key] = {"_name": name, "_count": 0, "_children": {}}
            node[key]["_count"] += 1
            node = node[key]["_children"]
    return tree


def main():
    text = README.read_text(encoding="utf-8")
    papers = parse_readme(text)

    # Deduplicate papers that appear in multiple sections by (title, url).
    # Keep first occurrence; record cross-references in `also_in`.
    seen = {}
    unique = []
    for p in papers:
        key = (p["title"].lower(), p.get("url") or "")
        if key in seen:
            idx = seen[key]
            other_path = unique[idx]["path"]
            if other_path != p["path"]:
                unique[idx].setdefault("also_in", []).append(p["path"])
            continue
        seen[key] = len(unique)
        unique.append(p)
    papers = unique

    taxonomy = build_taxonomy(papers)

    stats = {
        "total": len(papers),
        "with_code": sum(1 for p in papers if p["has_code"]),
        "new": sum(1 for p in papers if p["is_new"]),
        "systems": sum(1 for p in papers if p["venue_type"] == "systems"),
        "ai_ml": sum(1 for p in papers if p["venue_type"] == "ai-ml"),
        "preprint": sum(1 for p in papers if p["venue_type"] == "preprint"),
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
  --bg: #ffffff;
  --bg-soft: #f7f8fa;
  --bg-card: #ffffff;
  --border: #e5e7eb;
  --border-soft: #eef0f3;
  --text: #1f2328;
  --text-soft: #57606a;
  --text-mute: #8b949e;
  --accent: #6c5ce7;
  --accent-soft: #a29bfe;
  --accent-bg: #f3f1ff;
  --temporal: #3b82f6;
  --spatial: #10b981;
  --structural: #a855f7;
  --award: #d97706;
  --new: #ec4899;
  --link: #2563eb;
  --shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 12px rgba(15,23,42,.04);
  --shadow-hover: 0 4px 8px rgba(15,23,42,.06), 0 12px 28px rgba(15,23,42,.08);
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  font-size: 15px;
  line-height: 1.6;
  color: var(--text);
  background: var(--bg);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }

.container { max-width: 1200px; margin: 0 auto; padding: 0 24px; }

/* HERO */
.hero {
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, #fafbff 0%, #ffffff 100%);
  padding: 56px 0 40px;
}
.hero h1 {
  font-size: 36px;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 0 0 12px;
  background: linear-gradient(135deg, #6c5ce7 0%, #a855f7 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero .tagline { font-size: 17px; color: var(--text-soft); max-width: 720px; margin-bottom: 20px; }
.hero .meta { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; font-size: 14px; color: var(--text-soft); }
.hero .meta a { color: var(--text); font-weight: 500; }
.hero .meta .dot { color: var(--text-mute); }

.hero-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 16px;
  margin-top: 28px;
}
.hero-stat {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 18px;
  box-shadow: var(--shadow);
}
.hero-stat .num { font-size: 26px; font-weight: 700; color: var(--accent); line-height: 1.2; }
.hero-stat .label { font-size: 12px; color: var(--text-soft); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }

/* SECTION */
section { padding: 48px 0; }
section + section { border-top: 1px solid var(--border-soft); }
.section-title { font-size: 22px; font-weight: 700; margin: 0 0 6px; letter-spacing: -0.01em; }
.section-desc { color: var(--text-soft); font-size: 14px; margin: 0 0 24px; }

/* TAXONOMY */
.tax-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
.tax-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--bg-card);
  padding: 22px 22px 16px;
  box-shadow: var(--shadow);
  transition: transform .2s, box-shadow .2s;
  position: relative;
  overflow: hidden;
}
.tax-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-hover); }
.tax-card::before {
  content: "";
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: var(--bar-color, var(--accent));
}
.tax-card.temporal { --bar-color: var(--temporal); }
.tax-card.spatial { --bar-color: var(--spatial); }
.tax-card.structural { --bar-color: var(--structural); }

.tax-card .tax-icon { font-size: 22px; }
.tax-card h3 { font-size: 18px; font-weight: 700; margin: 8px 0 4px; }
.tax-card .tax-sub { font-size: 13px; color: var(--text-soft); margin-bottom: 14px; }
.tax-card .tax-count { font-size: 12px; color: var(--text-mute); font-weight: 600; }

.tax-children { margin-top: 12px; }
.tax-group { position: relative; }
.tax-group > .tax-child {
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 10px;
  border-radius: 6px;
  font-size: 13.5px;
  cursor: pointer;
  transition: background .15s;
}
.tax-group > .tax-child:hover { background: var(--bg-soft); }
.tax-group > .tax-child.active { background: var(--accent-bg); color: var(--accent); }
.tax-group > .tax-child.active .name { color: var(--accent); font-weight: 600; }
.tax-group > .tax-child.active .count { color: var(--accent); }
.tax-child .name { color: var(--text); font-weight: 500; }
.tax-child .count {
  color: var(--text-mute); font-size: 11.5px;
  background: var(--bg-soft);
  padding: 1px 7px; border-radius: 10px;
  font-weight: 600;
  min-width: 22px; text-align: center;
}
.tax-group > .tax-child.active .count {
  background: white; color: var(--accent);
}

.sub-children {
  margin: 2px 0 6px 14px;
  padding: 4px 0 4px 12px;
  border-left: 2px solid var(--border-soft);
}
.sub-children .tax-child {
  display: flex; justify-content: space-between; align-items: center;
  padding: 5px 10px;
  border-radius: 5px;
  font-size: 12.5px;
  cursor: pointer;
  transition: background .15s;
  color: var(--text-soft);
}
.sub-children .tax-child:hover { background: var(--bg-soft); }
.sub-children .tax-child.active { background: var(--accent-bg); }
.sub-children .tax-child.active .name { color: var(--accent); font-weight: 600; }
.sub-children .tax-child.active .count { background: white; color: var(--accent); }
.sub-children .tax-child .name { font-weight: 400; color: var(--text-soft); }
.sub-children .tax-child .name::before {
  content: "└ ";
  color: var(--text-mute);
  margin-right: 2px;
}
.sub-children .tax-child .count {
  font-size: 10.5px;
  background: transparent;
  padding: 1px 5px;
}

/* LANG TOGGLE */
.lang-toggle {
  position: absolute;
  top: 28px;
  right: 32px;
  display: inline-flex; align-items: center; gap: 5px;
  padding: 7px 14px;
  font-size: 13px;
  font-weight: 500;
  font-family: inherit;
  background: white;
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 18px;
  cursor: pointer;
  transition: all .15s;
  box-shadow: var(--shadow);
  z-index: 10;
}
.lang-toggle:hover {
  border-color: var(--accent);
  color: var(--accent);
  transform: translateY(-1px);
}
.lang-toggle .lang-icon { font-size: 14px; }

/* BROWSER */
.toolbar {
  display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  margin-bottom: 20px;
  padding: 16px;
  background: var(--bg-soft);
  border: 1px solid var(--border);
  border-radius: 10px;
}
.search {
  flex: 1 1 280px;
  display: flex; align-items: center;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0 12px;
}
.search input {
  flex: 1; border: 0; outline: 0;
  padding: 10px 8px; font-size: 14px; background: transparent;
  font-family: inherit; color: var(--text);
}
.search .icon { color: var(--text-mute); font-size: 16px; }

.filter-group { display: flex; gap: 6px; flex-wrap: wrap; }
.filter-group .label { font-size: 12px; color: var(--text-soft); margin-right: 4px; align-self: center; }
.chip {
  font-size: 12.5px;
  padding: 6px 12px;
  border-radius: 16px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-soft);
  cursor: pointer;
  transition: all .15s;
  user-select: none;
  white-space: nowrap;
}
.chip:hover { border-color: var(--accent-soft); color: var(--text); }
.chip.active { background: var(--accent); color: white; border-color: var(--accent); }

select.sort {
  padding: 7px 10px;
  font-size: 13px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-card);
  color: var(--text);
  cursor: pointer;
  font-family: inherit;
}

.result-bar {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 16px;
  font-size: 13px;
  color: var(--text-soft);
}
.result-bar .clear { color: var(--accent); cursor: pointer; margin-left: 12px; }
.result-bar .clear:hover { text-decoration: underline; }

.papers { display: flex; flex-direction: column; gap: 12px; }

.paper {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-card);
  padding: 18px 20px;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 16px;
  transition: border-color .15s, box-shadow .15s;
}
.paper:hover { border-color: var(--accent-soft); box-shadow: var(--shadow); }
.paper-main { min-width: 0; }
.paper-meta {
  display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
  margin-bottom: 6px;
}
.badge {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 11.5px; font-weight: 600;
  padding: 2px 8px; border-radius: 12px;
  letter-spacing: 0.02em;
  border: 1px solid transparent;
}
.badge.new { background: #fdf2f8; color: #be185d; border-color: #fbcfe8; }
.badge.systems { background: #ecfeff; color: #0e7490; border-color: #a5f3fc; }
.badge.ai-ml { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }
.badge.preprint { background: #f3f4f6; color: #4b5563; border-color: #d1d5db; }
.badge.award { background: #fffbeb; color: #b45309; border-color: #fde68a; }

.paper h4 { font-size: 15.5px; font-weight: 600; margin: 0 0 4px; line-height: 1.4; }
.paper h4 a { color: var(--text); }
.paper h4 a:hover { color: var(--link); }
.paper-authors { font-size: 13px; color: var(--text-soft); font-style: italic; margin: 0 0 8px; }
.paper-comment {
  font-size: 13px; color: var(--text-soft);
  background: var(--bg-soft); padding: 6px 10px; border-radius: 6px;
  border-left: 3px solid var(--accent-soft);
  margin-top: 6px;
}
.paper-path { font-size: 11.5px; color: var(--text-mute); margin-top: 8px; }
.paper-path .sep { color: var(--text-mute); margin: 0 4px; }
.paper-path .also { color: var(--accent); font-style: italic; }

.paper-code {
  display: flex; flex-direction: column; align-items: flex-end; gap: 4px;
  font-size: 12px; color: var(--text-soft);
  white-space: nowrap;
}
.paper-code a { display: inline-flex; align-items: center; gap: 4px; }
.paper-code .stars { font-weight: 600; color: var(--text); }
.paper-code .stars-icon { color: #f59e0b; }
.paper-code .repo { color: var(--link); font-family: ui-monospace, SFMono-Regular, monospace; font-size: 11.5px; }
.paper-code .no-code { color: var(--text-mute); font-size: 11.5px; }

.empty { padding: 40px 0; text-align: center; color: var(--text-mute); }

/* FOOTER */
footer {
  border-top: 1px solid var(--border);
  padding: 36px 0;
  font-size: 13px;
  color: var(--text-soft);
  background: var(--bg-soft);
}
footer .container { display: flex; flex-wrap: wrap; gap: 24px; justify-content: space-between; align-items: flex-start; }
footer h5 { font-size: 13px; font-weight: 700; margin: 0 0 8px; color: var(--text); }
footer pre {
  background: var(--bg-card); border: 1px solid var(--border);
  padding: 12px; border-radius: 6px;
  font-size: 11.5px; overflow-x: auto;
  font-family: ui-monospace, SFMono-Regular, monospace;
  max-width: 540px;
}

@media (max-width: 720px) {
  .hero h1 { font-size: 28px; }
  .hero { padding: 36px 0 24px; }
  .container { padding: 0 16px; }
  .paper { grid-template-columns: 1fr; }
  .paper-code { align-items: flex-start; flex-direction: row; }
  .lang-toggle { top: 20px; right: 16px; padding: 6px 12px; font-size: 12px; }
}
"""


BODY = """
<header class="hero">
  <div class="container" style="position: relative;">
    <button class="lang-toggle" id="lang-toggle" title="Switch language">
      <span class="lang-icon">🌐</span><span id="lang-label">中文</span>
    </button>
    <h1>Awesome KV Cache Optimization</h1>
    <p class="tagline" data-i18n="hero.tagline">A system-aware taxonomy of KV cache optimization methods for efficient LLM serving — covering temporal, spatial, and structural behaviors at serving time.</p>
    <div class="meta">
      <a href="https://aclanthology.org/2026.findings-acl.1916/" target="_blank">📄 ACL 2026 Findings</a>
      <span class="dot">·</span>
      <a href="https://github.com/jjiantong/Awesome-KV-Cache-Optimization" target="_blank">⭐ GitHub</a>
      <span class="dot">·</span>
      <a href="https://arxiv.org/abs/2607.08057" target="_blank">📥 arXiv</a>
    </div>
    <div class="hero-stats" id="hero-stats"></div>
  </div>
</header>

<section id="taxonomy">
  <div class="container">
    <h2 class="section-title" data-i18n="tax.title">Taxonomy at a Glance</h2>
    <p class="section-desc" data-i18n="tax.desc">A behavior-oriented taxonomy organizing system-aware KV cache optimization into three dimensions. Click any subcategory to filter the paper list below.</p>
    <div class="tax-grid" id="tax-grid"></div>
  </div>
</section>

<section id="papers">
  <div class="container">
    <h2 class="section-title" data-i18n="papers.title">Paper Explorer</h2>
    <p class="section-desc" id="papers-section-desc"></p>

    <div class="toolbar">
      <div class="search">
        <span class="icon">🔍</span>
        <input id="search" type="text" data-i18n-placeholder="papers.search_placeholder" placeholder="Search title, author, or technique..." autocomplete="off"/>
      </div>

      <div class="filter-group">
        <span class="label" data-i18n="filter.venue">Venue</span>
        <span class="chip active" data-filter="venue" data-value="all" data-i18n="chip.all">All</span>
        <span class="chip" data-filter="venue" data-value="systems" data-i18n="chip.systems">Systems</span>
        <span class="chip" data-filter="venue" data-value="ai-ml" data-i18n="chip.ai_ml">AI/ML</span>
        <span class="chip" data-filter="venue" data-value="preprint" data-i18n="chip.preprint">Preprint</span>
      </div>

      <div class="filter-group">
        <span class="label" data-i18n="filter.code">Code</span>
        <span class="chip active" data-filter="code" data-value="any" data-i18n="chip.any">Any</span>
        <span class="chip" data-filter="code" data-value="yes" data-i18n="chip.has_code">Has code</span>
        <span class="chip" data-filter="code" data-value="no" data-i18n="chip.no_code">No code</span>
      </div>

      <div class="filter-group">
        <span class="label" data-i18n="filter.more">More</span>
        <span class="chip" data-filter="new" data-value="yes" data-i18n="chip.new">🆕 New</span>
        <span class="chip" data-filter="award" data-value="yes" data-i18n="chip.award">🏆 Award</span>
      </div>

      <select class="sort" id="sort">
        <option value="default" data-i18n="sort.default">Sort: Default</option>
        <option value="newest" data-i18n="sort.newest">Sort: Newest venue</option>
        <option value="alpha" data-i18n="sort.alpha">Sort: A → Z</option>
        <option value="code-first" data-i18n="sort.code_first">Sort: Has code first</option>
      </select>
    </div>

    <div class="result-bar">
      <div>
        <span id="result-summary"></span>
        <span class="clear" id="clear-filters" style="display:none;" data-i18n="result.clear">✕ Clear filters</span>
      </div>
      <div id="active-cat" style="color: var(--accent);"></div>
    </div>

    <div class="papers" id="papers-list"></div>
  </div>
</section>

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
      'hero.tagline': 'A system-aware taxonomy of KV cache optimization methods for efficient LLM serving — covering temporal, spatial, and structural behaviors at serving time.',
      'stats.total': 'Papers',
      'stats.with_code': 'With Code',
      'stats.systems': 'Systems Venues',
      'stats.ai_ml': 'AI/ML Venues',
      'stats.preprint': 'Preprints',
      'stats.new': 'Newly Added',
      'stats.awarded': 'Awarded',
      'tax.title': 'Taxonomy at a Glance',
      'tax.desc': 'A behavior-oriented taxonomy organizing system-aware KV cache optimization into three dimensions. Click any subcategory to filter the paper list below.',
      'tax.count': '{n} papers',
      'tax.sub.temporal': 'when — execution & scheduling',
      'tax.sub.spatial': 'where — placement & migration',
      'tax.sub.structural': 'how — representation & retention',
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
      'chip.has_code': 'Has code',
      'chip.no_code': 'No code',
      'chip.new': '🆕 New',
      'chip.award': '🏆 Award',
      'sort.default': 'Sort: Default',
      'sort.newest': 'Sort: Newest venue',
      'sort.alpha': 'Sort: A → Z',
      'sort.code_first': 'Sort: Has code first',
      'result.summary': 'Showing {shown} of {total} papers',
      'result.clear': '✕ Clear filters',
      'result.empty': 'No papers match the current filters.',
      'result.empty_clear': 'Clear filters',
      'result.cat_label': '📂 {path}',
      'badge.new': '🆕 New',
      'badge.systems_default': 'Systems',
      'badge.ai_ml_default': 'AI/ML',
      'badge.preprint_default': 'Preprint',
      'paper.code': '📦 Code',
      'paper.no_code': '— no code —',
      'paper.also_in': '+ also in:',
      'footer.citation': 'Citation',
      'footer.cite_desc': 'If you find this resource helpful, please cite our survey:',
      'footer.links': 'Links',
      'footer.full_readme': '📖 Full README',
      'footer.note': 'Maintained by the ACL 2026 survey authors.<br/>Page generated from README.md via <code>build_page.py</code>.',
      'lang.switch_to': '中文',
    },
    zh: {
      'hero.tagline': '面向高效 LLM 服务的系统级 KV 缓存优化方法分类体系 —— 涵盖时间、空间、结构三种行为维度(仅涵盖服务期、无需重训的优化方法)。',
      'stats.total': '论文总数',
      'stats.with_code': '含开源代码',
      'stats.systems': '系统会议',
      'stats.ai_ml': 'AI/ML 会议',
      'stats.preprint': '预印本',
      'stats.new': '新近收录',
      'stats.awarded': '获奖论文',
      'tax.title': '分类体系总览',
      'tax.desc': '基于行为导向的分类体系,将系统级 KV 缓存优化组织为三大维度。点击任意子类即可筛选下方论文列表。',
      'tax.count': '{n} 篇论文',
      'tax.sub.temporal': '何时 —— 执行与调度',
      'tax.sub.spatial': '何地 —— 放置与迁移',
      'tax.sub.structural': '如何 —— 表示与保留',
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
      'chip.has_code': '有代码',
      'chip.no_code': '无代码',
      'chip.new': '🆕 新增',
      'chip.award': '🏆 获奖',
      'sort.default': '排序:默认',
      'sort.newest': '排序:最新会议',
      'sort.alpha': '排序:A → Z',
      'sort.code_first': '排序:有代码优先',
      'result.summary': '显示 {shown} / {total} 篇论文',
      'result.clear': '✕ 清除筛选',
      'result.empty': '没有论文匹配当前筛选条件。',
      'result.empty_clear': '清除筛选',
      'result.cat_label': '📂 {path}',
      'badge.new': '🆕 新增',
      'badge.systems_default': '系统',
      'badge.ai_ml_default': 'AI/ML',
      'badge.preprint_default': '预印本',
      'paper.code': '📦 代码',
      'paper.no_code': '— 暂无代码 —',
      'paper.also_in': '+ 还属于:',
      'footer.citation': '引用',
      'footer.cite_desc': '如果本资源对您有帮助,请引用我们的综述:',
      'footer.links': '相关链接',
      'footer.full_readme': '📖 完整 README',
      'footer.note': '由 ACL 2026 综述作者维护。<br/>本页面由 README.md 通过 <code>build_page.py</code> 自动生成。',
      'lang.switch_to': 'EN',
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
    // Lang toggle label
    document.getElementById('lang-label').textContent = t('lang.switch_to');
    // Section descriptions (with placeholders)
    const descEl = document.getElementById('papers-section-desc');
    if (descEl) {
      descEl.innerHTML = t('papers.desc', { total: '<strong>' + papers.length + '</strong>' });
    }
    // Re-render dynamic parts
    renderHeroStats();
    renderTaxGrid();
    renderResultBar();
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
      { num: s.ai_ml, label: t('stats.ai_ml') },
      { num: s.preprint, label: t('stats.preprint') },
      { num: s.new, label: t('stats.new') },
      { num: s.awarded, label: t('stats.awarded') },
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
  function behaviorIcon(name) {
    const n = name.toLowerCase();
    if (n.includes('temporal')) return '⏱️';
    if (n.includes('spatial')) return '💾';
    if (n.includes('structural')) return '🧩';
    return '📁';
  }
  function behaviorSubKey(name) {
    const n = name.toLowerCase();
    if (n.includes('temporal')) return 'tax.sub.temporal';
    if (n.includes('spatial')) return 'tax.sub.spatial';
    if (n.includes('structural')) return 'tax.sub.structural';
    return null;
  }

  function renderTaxNode(name, node) {
    const childKeys = Object.keys(node._children);
    if (!childKeys.length) return '';
    return '<div class="tax-children">' +
      childKeys.map(k => {
        const c = node._children[k];
        const subKeys = Object.keys(c._children);
        let sub2 = '';
        if (subKeys.length) {
          sub2 = '<div class="sub-children">' +
            subKeys.map(sk => {
              const sc = c._children[sk];
              return `<div class="tax-child" data-cat="${escapeAttr(name)}|${escapeAttr(c._name)}|${escapeAttr(sk)}">
                <span class="name">${escapeHtml(translateCat(sc._name))}</span>
                <span class="count">${sc._count}</span>
              </div>`;
            }).join('') +
          '</div>';
        }
        return `<div class="tax-group">
          <div class="tax-child" data-cat="${escapeAttr(name)}|${escapeAttr(c._name)}">
            <span class="name">${escapeHtml(translateCat(c._name))}</span>
            <span class="count">${c._count}</span>
          </div>
          ${sub2}
        </div>`;
      }).join('') +
    '</div>';
  }

  function renderTaxCard(name, node) {
    const cls = behaviorClass(name);
    const subKey = behaviorSubKey(name);
    const subText = subKey ? t(subKey) : '';
    return `<div class="tax-card ${cls}">
      <div class="tax-icon">${behaviorIcon(name)}</div>
      <h3>${escapeHtml(translateCat(name))}</h3>
      <div class="tax-sub">${escapeHtml(subText)}</div>
      <div class="tax-count">${escapeHtml(t('tax.count', { n: node._count }))}</div>
      ${renderTaxNode(name, node)}
    </div>`;
  }

  function renderTaxGrid() {
    const taxGrid = document.getElementById('tax-grid');
    const topKeys = Object.keys(DATA.taxonomy);
    taxGrid.innerHTML = topKeys.map(k => renderTaxCard(k, DATA.taxonomy[k])).join('');
    // Restore active state if cat is set
    if (state.cat) {
      const el = taxGrid.querySelector(`.tax-child[data-cat="${cssAttrEscape(state.cat)}"]`);
      if (el) el.classList.add('active');
    }
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
    });
  }

  function clearCategory() {
    state.cat = '';
    document.querySelectorAll('.tax-child.active').forEach(el => el.classList.remove('active'));
    document.getElementById('active-cat').textContent = '';
  }

  function cssAttrEscape(s) {
    return String(s).replace(/"/g, '\\"');
  }

  function renderCategoryBreadcrumb() {
    const el = document.getElementById('active-cat');
    if (!state.cat) { el.textContent = ''; return; }
    const parts = state.cat.split('|').map(translateCat);
    el.textContent = t('result.cat_label', { path: parts.join(' › ') });
  }

  function renderResultBar() {
    renderCategoryBreadcrumb();
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

  // Tax click (delegated)
  document.getElementById('tax-grid').addEventListener('click', (e) => {
    const el = e.target.closest('.tax-child');
    if (!el) return;
    const cat = el.dataset.cat;
    if (state.cat === cat) {
      clearCategory();
    } else {
      document.querySelectorAll('.tax-child.active').forEach(n => n.classList.remove('active'));
      el.classList.add('active');
      state.cat = cat;
      renderCategoryBreadcrumb();
    }
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
    if (state.cat) {
      const parts = state.cat.split('|');
      const pathJoined = p.path.join('|');
      const prefix = parts.join('|');
      if (!(pathJoined === prefix || pathJoined.startsWith(prefix + '|'))) return false;
    }
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
    } else {
      return `<span class="badge preprint">📄 ${escapeHtml(v)}</span>`;
    }
  }

  function renderPaper(p) {
    const badges = [];
    if (p.is_new) badges.push(`<span class="badge new">${t('badge.new')}</span>`);
    badges.push(venueBadge(p));
    if (p.award) badges.push(`<span class="badge award">🏆 ${escapeHtml(p.award)}</span>`);

    let codeHtml;
    if (p.has_code && p.github_owner && p.github_repo) {
      codeHtml = `
        <a href="${escapeAttr(p.github_url)}" target="_blank" rel="noopener" title="GitHub stars">
          <span class="stars-icon">★</span>
          <span class="stars" data-stars="${escapeAttr(p.github_owner)}/${escapeAttr(p.github_repo)}">…</span>
        </a>
        <a href="${escapeAttr(p.github_url)}" target="_blank" rel="noopener" class="repo">${escapeHtml(p.github_repo)}</a>`;
    } else if (p.has_code && p.github_url) {
      codeHtml = `<a href="${escapeAttr(p.github_url)}" target="_blank" rel="noopener">${t('paper.code')}</a>`;
    } else {
      codeHtml = `<span class="no-code">${escapeHtml(t('paper.no_code'))}</span>`;
    }

    let pathHtml = p.path.map((name, i) => {
      const dispName = translateCat(name);
      if (i === 0) {
        const cls = behaviorClass(name);
        const color = cls === 'temporal' ? 'var(--temporal)' :
                      cls === 'spatial' ? 'var(--spatial)' :
                      cls === 'structural' ? 'var(--structural)' : 'var(--text-mute)';
        return `<span style="color:${color};font-weight:600">${escapeHtml(dispName)}</span>`;
      }
      return `<span>${escapeHtml(dispName)}</span>`;
    }).join('<span class="sep">›</span>');

    if (p.also_in && p.also_in.length) {
      const alsoText = t('paper.also_in');
      const alsoPaths = p.also_in.map(a => a.map(translateCat).join('<span class="sep">›</span>')).join(', ');
      pathHtml += ` <span class="also">${escapeHtml(alsoText)} ${alsoPaths}</span>`;
    }

    const titleHtml = p.url
      ? `<h4><a href="${escapeAttr(p.url)}" target="_blank" rel="noopener">${escapeHtml(p.title)}</a></h4>`
      : `<h4>${escapeHtml(p.title)}</h4>`;

    return `<article class="paper">
      <div class="paper-main">
        <div class="paper-meta">${badges.join('')}</div>
        ${titleHtml}
        ${p.authors ? `<div class="paper-authors">${escapeHtml(p.authors)}</div>` : ''}
        ${p.comment ? `<div class="paper-comment">${escapeHtml(p.comment)}</div>` : ''}
        <div class="paper-path">${pathHtml}</div>
      </div>
      <div class="paper-code">${codeHtml}</div>
    </article>`;
  }

  function render() {
    const filtered = sortPapers(papers.filter(matches));
    const list = document.getElementById('papers-list');
    if (filtered.length === 0) {
      const emptyClear = `<span class="clear" id="empty-clear">${escapeHtml(t('result.empty_clear'))}</span>`;
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
    document.getElementById('clear-filters').style.display =
      (state.search || state.venue !== 'all' || state.code !== 'any' || state.new || state.award || state.cat) ? 'inline' : 'none';
    fetchStars();
  }

  // ---- LAZY STAR FETCH ----
  let starsFetched = false;
  function fetchStars() {
    if (starsFetched) return;
    starsFetched = true;
    document.querySelectorAll('[data-stars]').forEach(el => {
      const repo = el.dataset.stars;
      const url = `https://api.github.com/repos/${repo}`;
      fetch(url).then(r => r.json()).then(d => {
        if (typeof d.stargazers_count === 'number') {
          el.textContent = formatStars(d.stargazers_count);
        } else {
          el.textContent = '★';
        }
      }).catch(() => { el.textContent = '★'; });
    });
  }

  function formatStars(n) {
    if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
    return String(n);
  }

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
