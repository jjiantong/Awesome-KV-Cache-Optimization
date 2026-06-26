# --------------------------------------------
# User Input Section
# --------------------------------------------

# Please fill in the title and authors
paper_name = ""
author_name = ""

# Please provide the paper link
paper_url = ""

# Please provide the GitHub link (if available) and 
# manually provide its number of stars
github_url = ""
stars_manual = 0

# Please fill in the publication venue (if available)
# For conferences, use the format: Conference-ICLR_2025
# For journals, use the format: Journal-JMLR_2025
# Leave blank and ignore the next question if it's an arXiv preprint
venue = ""

# Please specify whether this is a systems venue (1) or AI/ML venue (0)
# 1 = Systems (OSDI, NSDI, MLSys, ASPLOS, etc.)
# 0 = AI/ML or general venue (ICLR, NeurIPS, ICML, JMLR, etc.)
venue_type = 0

# Optional comment for the second column
comment = ""


# --------------------------------------------
# Logic Section
# --------------------------------------------

def parse_github_url(url: str):
    if "github.com" not in url:
        return None, None
    parts = url.split("github.com/")[-1].strip("/")
    if len(parts.split("/")) < 2:
        return None, None
    owner, repo = parts.split("/")[:2]
    return owner, repo

def make_badge(venue: str, venue_type: int):
    if not venue or "arxiv" in venue.lower():
        return ""
    color = "cyan" if venue_type == 1 else "blue"
    return f"[![Publish](https://img.shields.io/badge/{venue}-{color})]() <br>"

# --------------------------------------------
# Build Markdown
# --------------------------------------------
venue_badge = make_badge(venue, venue_type)
paper_title = f"{paper_name} [[Link]({paper_url})]" if paper_url else paper_name
author_block = f"*{author_name}*" if author_name else ""
left_col = f"{venue_badge} {paper_title} <br> {author_block}".strip()

owner, repo = parse_github_url(github_url)
if owner and repo:
    star_badge = f"![](https://img.shields.io/github/stars/{owner}/{repo}?style=social)"
    commit_badge = f"![](https://img.shields.io/github/last-commit/{owner}/{repo}?color=green)"
    repo_link = f"[{repo}](https://github.com/{owner}/{repo})"
    star_emoji = " 🌟" if stars_manual >= 1000 else ""
    right_col = f"{star_badge} <br> {commit_badge} <br> {repo_link}{star_emoji}"
else:
    right_col = ""

markdown_row = f"| {left_col} | {comment} | {right_col} |"

print("\n✅ Generated Markdown row:\n")
print(markdown_row)