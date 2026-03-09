"""Generate weekly Beamer slides from daily progress reports.

Parses daily markdown reports, TASKS.md, and figures/ to produce
a LaTeX Beamer slide deck compiled to PDF.

Usage:
    python automation/generate_weekly_slides.py                  # current week
    python automation/generate_weekly_slides.py 2026-02-10       # week containing this date
    python automation/generate_weekly_slides.py 2026-02-10 2026-02-16  # explicit range
"""

from __future__ import annotations

import os
import re
import sys
import glob
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# ── Paths ─────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_DIR / "reports"
WEEKLY_DIR = REPORTS_DIR / "weekly"
FIGURES_DIR = PROJECT_DIR / "figures"
TASKS_FILE = PROJECT_DIR / "TASKS.md"
TEMPLATE_FILE = PROJECT_DIR / "automation" / "templates" / "weekly_beamer.tex"


# ── LaTeX escaping ────────────────────────────────────────────────────────

_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Regex: match any special char NOT already preceded by backslash
_LATEX_ESCAPE_RE = re.compile(
    r"(?<!\\)([" + re.escape("".join(_LATEX_SPECIAL.keys())) + r"])"
)


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters, preserving intentional LaTeX commands."""
    # First pass: escape special chars
    result = _LATEX_ESCAPE_RE.sub(lambda m: _LATEX_SPECIAL[m.group(1)], text)
    # Fix common patterns that should stay as-is
    result = result.replace(r"\\_", r"\_")  # already escaped underscores
    # Handle < and > for comparisons (p < 0.001, > 25)
    result = re.sub(r"(?<!\$)<(?!\$)", r"$<$", result)
    result = re.sub(r"(?<!\$)>(?!\$)", r"$>$", result)
    # Restore markdown bold -> LaTeX bold
    result = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", result)
    return result


def md_to_latex_inline(text: str) -> str:
    """Convert inline markdown formatting to LaTeX."""
    text = escape_latex(text)
    # backtick code -> \texttt
    text = re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)
    return text


# ── Markdown parsing ──────────────────────────────────────────────────────

def parse_daily_report(filepath: Path) -> dict:
    """Parse a daily report markdown file into sections."""
    content = filepath.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    current_section = "preamble"
    lines: list[str] = []

    for line in content.splitlines():
        if line.startswith("## "):
            if lines:
                sections[current_section] = "\n".join(lines)
            current_section = line[3:].strip()
            lines = []
        else:
            lines.append(line)

    if lines:
        sections[current_section] = "\n".join(lines)

    return sections


def extract_bullets(text: str) -> list[str]:
    """Extract top-level bullet points from markdown text."""
    bullets = []
    for line in text.splitlines():
        m = re.match(r"^[-*]\s+(.+)", line)
        if not m:
            m = re.match(r"^- \[.\]\s+(.+)", line)
        if m:
            bullets.append(m.group(1).strip())
    return bullets


def extract_task_bullets(text: str) -> list[str]:
    """Extract completed task bullets, including sub-headers as context."""
    bullets = []
    for line in text.splitlines():
        m = re.match(r"^- \[x\]\s+(.+)", line)
        if m:
            task_text = m.group(1).strip()
            task_text = _sentence_truncate(task_text, max_len=120)
            bullets.append(task_text)
    return bullets


def _sentence_truncate(text: str, max_len: int = 150) -> str:
    """Truncate text at a sentence boundary, avoiding decimal point cuts."""
    if len(text) <= max_len:
        return text
    # Find sentence-ending periods: period followed by space (not digit)
    for m in re.finditer(r"\.(?:\s|$)", text):
        if m.start() >= 50 and m.start() <= max_len:
            return text[:m.start() + 1]
    # Fallback: truncate at word boundary
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def extract_findings(text: str) -> list[str]:
    """Extract key findings as bullet points, preferring bold-labeled items."""
    findings = []
    for line in text.splitlines():
        # Skip table rows and separator lines
        if line.strip().startswith("|") or line.strip().startswith("---"):
            continue
        m = re.match(r"^[-*]\s+(.+)", line)
        if m:
            finding = m.group(1).strip()
            # Skip short/trivial lines
            if len(finding) < 25:
                continue
            # Prefer lines with bold labels or quantitative results
            has_bold = "**" in finding
            has_number = bool(re.search(r"\d+\.\d+", finding))
            if not (has_bold or has_number):
                continue
            finding = _sentence_truncate(finding)
            findings.append(finding)
    return findings


def extract_issues(text: str) -> list[str]:
    """Extract issues/blockers as bullet points, keeping them concise."""
    issues = []
    for line in text.splitlines():
        m = re.match(r"^[-*]\s+\*\*(.+?)\*\*:?\s*(.*)", line)
        if m:
            title = m.group(1).strip()
            desc = m.group(2).strip()
            # Truncate description to first sentence
            if desc:
                sent = re.match(r"^(.+?\.)\s", desc)
                desc = sent.group(1) if sent else desc[:120]
            issues.append(f"**{title}**: {desc}" if desc else f"**{title}**")
        else:
            m = re.match(r"^[-*]\s+(.+)", line)
            if m:
                item = m.group(1).strip()
                if len(item) > 140:
                    item = item[:140].rsplit(" ", 1)[0] + "..."
                issues.append(item)
    return issues


# ── TASKS.md parsing ──────────────────────────────────────────────────────

def parse_phase_progress(tasks_file: Path) -> list[dict]:
    """Parse TASKS.md to get per-phase completion counts."""
    content = tasks_file.read_text(encoding="utf-8")
    phases = []
    current_phase = None
    done = 0
    total = 0

    for line in content.splitlines():
        m = re.match(r"^## Phase (\d+):\s*(.+?)(?:\s*\(.*\))?\s*$", line)
        if m:
            if current_phase is not None:
                phases.append({
                    "num": current_phase["num"],
                    "name": current_phase["name"],
                    "done": done,
                    "total": total,
                })
            current_phase = {"num": int(m.group(1)), "name": m.group(2).strip()}
            done = 0
            total = 0
            continue

        if current_phase is not None:
            if re.match(r"^- \[x\]", line):
                done += 1
                total += 1
            elif re.match(r"^- \[[ ~!]\]", line):
                total += 1

    if current_phase is not None:
        phases.append({
            "num": current_phase["num"],
            "name": current_phase["name"],
            "done": done,
            "total": total,
        })

    return phases


# ── Figure scanning ───────────────────────────────────────────────────────

def find_week_figures(
    figures_dir: Path, start_date: datetime, end_date: datetime
) -> list[Path]:
    """Find publication PNG figures created/modified during the week.

    Only includes files matching 'fig*_*.png' (publication figures)
    and caps at 12 figures to keep slides manageable.
    """
    figures = []
    if not figures_dir.exists():
        return figures
    # Only publication figures (fig1_*, fig2_*, ..., fig12_*)
    for f in sorted(figures_dir.glob("fig*_*.png")):
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        if start_date <= mtime <= end_date + timedelta(days=1):
            figures.append(f)
    return figures[:12]


# ── LaTeX content generators ─────────────────────────────────────────────

def _first_sentence(text: str, max_len: int = 180) -> str:
    """Extract the first sentence from text, truncated to max_len."""
    text = text.strip().splitlines()[0].strip() if text.strip() else ""
    # Split at first period followed by space or end
    m = re.match(r"^(.+?\.)\s", text)
    sentence = m.group(1) if m else text
    if len(sentence) > max_len:
        sentence = sentence[:max_len].rsplit(" ", 1)[0] + "..."
    return sentence


def gen_executive_summary(reports: list[dict]) -> str:
    """Generate executive summary from daily Session Summary sections."""
    summaries = []
    for r in reports:
        text = r.get("Session Summary", "")
        sentence = _first_sentence(text)
        if sentence:
            summaries.append(sentence)

    if not summaries:
        return "\\begin{itemize}\n  \\item No reports found for this week.\n\\end{itemize}"

    lines = ["\\begin{itemize}\\small"]
    for s in summaries:
        lines.append(f"  \\item {md_to_latex_inline(s)}")
    lines.append("\\end{itemize}")
    return "\n".join(lines)


def gen_phase_progress(phases: list[dict]) -> str:
    """Generate tabular progress bars for each phase."""
    rows = []
    for p in phases:
        frac = p["done"] / p["total"] if p["total"] > 0 else 0
        label = f"{p['done']}/{p['total']}"
        name = escape_latex(f"Phase {p['num']}: {p['name']}")
        rows.append(f"\\phasebar{{{name}}}{{{frac:.2f}}}{{{label}}}")
    inner = "\n".join(rows)
    return (
        "\\begin{tabular}{@{}l@{\\hspace{8pt}}c@{\\hspace{8pt}}r@{}}\n"
        f"{inner}"
        "\n\\end{tabular}"
    )


def gen_tasks_completed(reports: list[dict]) -> str:
    """Generate Tasks Completed frames (1-2 frames depending on count)."""
    all_tasks: list[str] = []
    for r in reports:
        text = r.get("Tasks Completed", "")
        tasks = extract_task_bullets(text)
        all_tasks.extend(tasks)

    if not all_tasks:
        return (
            "\\begin{frame}{Tasks Completed}\n"
            "  \\centering No tasks completed this week.\n"
            "\\end{frame}"
        )

    # Split into frames of max 8 items
    frames = []
    chunk_size = 8
    for i in range(0, len(all_tasks), chunk_size):
        chunk = all_tasks[i : i + chunk_size]
        suffix = f" ({i // chunk_size + 1})" if len(all_tasks) > chunk_size else ""
        items = "\n".join(f"  \\item {md_to_latex_inline(t)}" for t in chunk)
        frames.append(
            f"\\begin{{frame}}{{Tasks Completed{suffix}}}\n"
            f"\\begin{{itemize}}\\small\n{items}\n\\end{{itemize}}\n"
            f"\\end{{frame}}"
        )
    return "\n\n".join(frames)


def gen_key_findings(reports: list[dict]) -> str:
    """Generate Key Findings frames.

    Prioritizes later reports (modeling/analysis) over earlier ones
    (preprocessing stats) by processing reports in reverse order.
    """
    all_findings: list[str] = []
    # Reverse: later reports have higher-impact findings
    for r in reversed(reports):
        text = r.get("Key Findings / Observations", "")
        findings = extract_findings(text)
        all_findings.extend(findings)

    if not all_findings:
        return (
            "\\begin{frame}{Key Findings}\n"
            "  \\centering No key findings this week.\n"
            "\\end{frame}"
        )

    # Deduplicate
    seen = set()
    unique: list[str] = []
    for f in all_findings:
        key = f[:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)

    # Limit to 12 findings max (cap at 2 frames)
    unique = unique[:12]

    frames = []
    chunk_size = 6
    for i in range(0, len(unique), chunk_size):
        chunk = unique[i : i + chunk_size]
        suffix = f" ({i // chunk_size + 1})" if len(unique) > chunk_size else ""
        items = "\n".join(f"  \\item {md_to_latex_inline(f)}" for f in chunk)
        frames.append(
            f"\\begin{{frame}}{{Key Findings{suffix}}}\n"
            f"\\begin{{itemize}}\\small\n{items}\n\\end{{itemize}}\n"
            f"\\end{{frame}}"
        )
    return "\n\n".join(frames)


def gen_figures_block(figures: list[Path]) -> str:
    """Generate figure inclusion frames."""
    if not figures:
        return "% No new figures this week."

    frames = []
    # Group into pairs for 2-column layout
    for i in range(0, len(figures), 4):
        chunk = figures[i : i + 4]
        suffix = f" ({i // 4 + 1})" if len(figures) > 4 else ""

        inner = []
        if len(chunk) <= 2:
            # Side by side
            cols = []
            for fig in chunk:
                relpath = os.path.relpath(fig, PROJECT_DIR).replace("\\", "/")
                name = fig.stem.replace("_", r"\_")
                cols.append(
                    f"    \\begin{{column}}{{0.48\\textwidth}}\n"
                    f"      \\centering\n"
                    f"      \\includegraphics[width=\\linewidth,height=0.65\\textheight,keepaspectratio]{{{relpath}}}\n"
                    f"      \\\\\\tiny {name}\n"
                    f"    \\end{{column}}"
                )
            inner.append(
                "  \\begin{columns}\n" + "\n".join(cols) + "\n  \\end{columns}"
            )
        else:
            # 2x2 grid
            rows = []
            for j in range(0, len(chunk), 2):
                pair = chunk[j : j + 2]
                cols = []
                for fig in pair:
                    relpath = os.path.relpath(fig, PROJECT_DIR).replace("\\", "/")
                    name = fig.stem.replace("_", r"\_")
                    cols.append(
                        f"      \\begin{{column}}{{0.48\\textwidth}}\n"
                        f"        \\centering\n"
                        f"        \\includegraphics[width=\\linewidth,height=0.3\\textheight,keepaspectratio]{{{relpath}}}\n"
                        f"        \\\\\\tiny {name}\n"
                        f"      \\end{{column}}"
                    )
                rows.append(
                    "    \\begin{columns}\n" + "\n".join(cols) + "\n    \\end{columns}"
                )
            inner.append("\n  \\vspace{4pt}\n".join(rows))

        frames.append(
            f"\\begin{{frame}}{{New Figures{suffix}}}\n"
            + "\n".join(inner)
            + "\n\\end{frame}"
        )

    return "\n\n".join(frames)


def gen_issues_blockers(reports: list[dict]) -> str:
    """Generate Issues & Blockers content."""
    all_issues: list[str] = []
    for r in reports:
        text = r.get("Issues / Blockers", "")
        issues = extract_issues(text)
        all_issues.extend(issues)

    if not all_issues:
        return "\\centering No issues or blockers this week."

    # Deduplicate
    seen = set()
    unique: list[str] = []
    for issue in all_issues:
        key = issue[:50].lower()
        if key not in seen:
            seen.add(key)
            unique.append(issue)

    items = "\n".join(f"  \\item {md_to_latex_inline(i)}" for i in unique[:8])
    return f"\\begin{{itemize}}\\small\n{items}\n\\end{{itemize}}"


def gen_next_week_plan(reports: list[dict]) -> str:
    """Generate Next Week Plan from the most recent report."""
    for r in reversed(reports):
        text = r.get("Next Session Plan", "")
        bullets = extract_bullets(text)
        if bullets:
            items = "\n".join(f"  \\item {md_to_latex_inline(b)}" for b in bullets)
            return f"\\begin{{enumerate}}\n{items}\n\\end{{enumerate}}"

    return "\\centering Plan to be determined."


# ── Main generator ────────────────────────────────────────────────────────

def determine_week_range(
    arg1: Optional[str] = None, arg2: Optional[str] = None
) -> tuple[datetime, datetime]:
    """Determine the week date range from CLI arguments."""
    if arg1 and arg2:
        start = datetime.strptime(arg1, "%Y-%m-%d")
        end = datetime.strptime(arg2, "%Y-%m-%d")
    elif arg1:
        ref = datetime.strptime(arg1, "%Y-%m-%d")
        start = ref - timedelta(days=ref.weekday())  # Monday
        end = start + timedelta(days=6)  # Sunday
    else:
        today = datetime.now()
        start = today - timedelta(days=today.weekday())  # Monday
        end = start + timedelta(days=6)
    return start, end


def find_daily_reports(start: datetime, end: datetime) -> list[Path]:
    """Find daily report files within the date range."""
    reports = []
    current = start
    while current <= end:
        fname = REPORTS_DIR / f"{current.strftime('%Y-%m-%d')}.md"
        if fname.exists():
            reports.append(fname)
        current += timedelta(days=1)
    return sorted(reports)


def generate(start: datetime, end: datetime) -> Path:
    """Generate the weekly Beamer slides and compile to PDF."""
    WEEKLY_DIR.mkdir(parents=True, exist_ok=True)

    # Week label
    iso_year, iso_week, _ = start.isocalendar()
    week_short = f"{iso_year}-W{iso_week:02d}"
    date_range = f"{start.strftime('%b %d')}" + "--" + f"{end.strftime('%d, %Y')}"

    print(f"Generating weekly slides: {week_short} ({date_range})")

    # Find and parse daily reports
    report_files = find_daily_reports(start, end)
    print(f"  Found {len(report_files)} daily reports")
    if not report_files:
        print("  WARNING: No daily reports found for this week.")

    parsed_reports = [parse_daily_report(f) for f in report_files]

    # Parse phase progress
    phases = parse_phase_progress(TASKS_FILE) if TASKS_FILE.exists() else []
    print(f"  Parsed {len(phases)} phases from TASKS.md")

    # Find figures
    figures = find_week_figures(FIGURES_DIR, start, end)
    print(f"  Found {len(figures)} figures from this week")

    # Load template
    template = TEMPLATE_FILE.read_text(encoding="utf-8")

    # Generate content blocks
    replacements = {
        "<<WEEK_SHORT>>": escape_latex(week_short),
        "<<DATE_RANGE>>": escape_latex(date_range),
        "<<EXECUTIVE_SUMMARY>>": gen_executive_summary(parsed_reports),
        "<<PHASE_PROGRESS>>": gen_phase_progress(phases),
        "<<TASKS_COMPLETED>>": gen_tasks_completed(parsed_reports),
        "<<KEY_FINDINGS>>": gen_key_findings(parsed_reports),
        "<<FIGURES_BLOCK>>": gen_figures_block(figures),
        "<<ISSUES_BLOCKERS>>": gen_issues_blockers(parsed_reports),
        "<<NEXT_WEEK_PLAN>>": gen_next_week_plan(parsed_reports),
    }

    # Apply replacements
    tex_content = template
    for token, value in replacements.items():
        tex_content = tex_content.replace(token, value)

    # Write .tex
    tex_path = WEEKLY_DIR / f"{week_short}.tex"
    tex_path.write_text(tex_content, encoding="utf-8")
    print(f"  Written: {tex_path}")

    # Compile with pdflatex (two passes for references)
    pdf_path = WEEKLY_DIR / f"{week_short}.pdf"
    for pass_num in (1, 2):
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory", str(WEEKLY_DIR),
                str(tex_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_DIR),
            timeout=120,
        )
        if result.returncode != 0 and pass_num == 2:
            print(f"  WARNING: pdflatex returned {result.returncode}")
            # Write log for debugging
            log_path = WEEKLY_DIR / f"{week_short}_compile.log"
            log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
            print(f"  Compile log: {log_path}")

    if pdf_path.exists():
        print(f"  PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)")
    else:
        print("  ERROR: PDF not generated. Check compile log.")

    # Clean aux files
    for ext in (".aux", ".log", ".nav", ".out", ".snm", ".toc", ".vrb"):
        aux = WEEKLY_DIR / f"{week_short}{ext}"
        if aux.exists():
            aux.unlink()

    return pdf_path


def main() -> None:
    args = sys.argv[1:]
    arg1 = args[0] if len(args) >= 1 else None
    arg2 = args[1] if len(args) >= 2 else None
    start, end = determine_week_range(arg1, arg2)
    generate(start, end)


if __name__ == "__main__":
    main()
