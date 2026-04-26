#!/usr/bin/env python3
"""
Produce two archives from JMIV2026_rev1/paper.tex:

  out/JMIV_zipped/       — files as-is, \replaced/\added/\deleted visible
  out/JMIV_zipped.zip

  out/JMIV_zipped_clean/ — .tex/.tikz stripped of changes markup
  out/JMIV_zipped_clean.zip
"""

import re
import shutil
from pathlib import Path

SOURCE  = Path("/home/romain22222/visibilityLattices/JMIV2026_rev1")
OUT_DIR = Path("/home/romain22222/visibilityLattices/out/JMIV_zipped")
OUT_CLEAN = Path("/home/romain22222/visibilityLattices/out/JMIV_zipped_clean")

IMAGE_EXTS = [".pdf", ".png", ".jpg", ".jpeg", ".eps"]

# ── LaTeX brace-aware helpers ─────────────────────────────────────────────────

def extract_brace_group(text: str, start: int) -> tuple[str, int]:
    """Return (content, pos_after_closing_brace) for the {}-group at `start`."""
    assert text[start] == "{"
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return text[start + 1 :], len(text)


def skip_optional(text: str, pos: int) -> int:
    """Skip whitespace then an optional [...] argument if present."""
    while pos < len(text) and text[pos] in " \t\n":
        pos += 1
    if pos < len(text) and text[pos] == "[":
        depth = 0
        while pos < len(text):
            if text[pos] == "[":
                depth += 1
            elif text[pos] == "]":
                depth -= 1
                if depth == 0:
                    return pos + 1
            pos += 1
    return pos


def skip_brace_group(text: str, pos: int) -> int:
    """Skip whitespace then a {}-group; return position after it."""
    while pos < len(text) and text[pos] in " \t\n":
        pos += 1
    if pos < len(text) and text[pos] == "{":
        _, pos = extract_brace_group(text, pos)
    return pos


# Commands that must be REMOVED entirely (no content kept)
_REMOVE_CMDS = re.compile(
    r"\\(?:definechangesauthor|setdeletedmarkup|listofchanges)"
    r"(?:\[[^\]]*\])?"   # optional [...]
    r"(?:\{[^}]*\})*"    # zero or more {}-groups (simple, non-nested)
)

# \usepackage[...]{changes}  → replaced with [final] variant (keeps pkg for class deps)
_REMOVE_PKG = re.compile(r"(\\usepackage)(?:\[[^\]]*\])?\{changes\}")


def strip_changes(text: str) -> str:
    """
    Replace changes-package markup with clean LaTeX:
      \\replaced[...]{NEW}{OLD}  →  NEW
      \\added[...]{TEXT}         →  TEXT
      \\deleted[...]{TEXT}       →  (empty)
    Also removes \\definechangesauthor, \\setdeletedmarkup,
    \\newcommand{\\stkout}, and switches changes to [final] mode.
    """
    # 0. Remove helper definitions BEFORE char-by-char pass to avoid
    #    misidentifying \stkout inside \newcommand{\stkout} as a call
    text = re.sub(r"\\newcommand\{\\stkout\}[^\n]*\n?", "", text)
    text = re.sub(r"\\setdeletedmarkup[^\n]*\n?",       "", text)

    # 1. Switch changes package to [final] — keeps sn-jnl.cls dependencies intact
    text = _REMOVE_PKG.sub(r"\1[final]{changes}", text)

    result: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] != "\\":
            result.append(text[i])
            i += 1
            continue

        # Try to match one of our target commands
        handled = False
        for cmd, action in (
            ("replaced",            "keep_first_skip_second"),
            ("added",               "keep_first"),
            ("deleted",             "skip_first"),
            ("definechangesauthor", "skip_all"),
        ):
            end_of_name = i + 1 + len(cmd)
            if text[i + 1 : end_of_name] == cmd and (
                end_of_name >= n or not text[end_of_name].isalpha()
            ):
                pos = end_of_name
                pos = skip_optional(text, pos)        # skip [id=X]
                while pos < n and text[pos] in " \t\n":
                    pos += 1

                if action == "skip_all":
                    pos = skip_optional(text, pos)
                    while pos < n and text[pos] == "{":
                        pos = skip_brace_group(text, pos)
                    i = pos

                elif action in ("keep_first", "keep_first_skip_second"):
                    if pos < n and text[pos] == "{":
                        first, pos = extract_brace_group(text, pos)
                        result.append(first)
                        if action == "keep_first_skip_second":
                            pos = skip_brace_group(text, pos)
                    i = pos

                elif action == "skip_first":
                    if pos < n and text[pos] == "{":
                        _, pos = extract_brace_group(text, pos)
                    i = pos

                handled = True
                break

        if not handled:
            result.append(text[i])
            i += 1

    return "".join(result)


# ── file collection ───────────────────────────────────────────────────────────

def resolve(ref: str, exts=None) -> Path | None:
    p = SOURCE / ref
    if p.exists():
        return p
    for ext in (exts or []):
        c = SOURCE / (ref + ext)
        if c.exists():
            return c
    return None


def collect(tex_path: Path, visited: set, files: set):
    if tex_path in visited:
        return
    visited.add(tex_path)

    if not tex_path.exists():
        with_tex = tex_path.with_suffix(".tex")
        if with_tex.exists():
            tex_path = with_tex
        else:
            print(f"  [WARN] not found: {tex_path}")
            return

    files.add(tex_path)
    text = tex_path.read_text(errors="replace")

    for m in re.finditer(r"\\(?:input|include)\s*\{([^}]+)\}", text):
        ref = m.group(1).strip()
        cand = SOURCE / ref
        if not cand.suffix:
            cand = cand.with_suffix(".tex")
        collect(cand, visited, files)

    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}", text):
        ref = m.group(1).strip()
        if "#" in ref:
            continue
        p = resolve(ref, IMAGE_EXTS)
        if p:
            files.add(p)
        else:
            print(f"  [WARN] image not found: {ref}")

    for m in re.finditer(r"\]\s*\{([^}]+\.txt)\}", text):
        ref = m.group(1).strip()
        p = resolve(ref)
        if p:
            files.add(p)
        else:
            print(f"  [WARN] data file not found: {ref}")

    for m in re.finditer(r"\\bibliography\s*\{([^}]+)\}", text):
        p = resolve(m.group(1).strip() + ".bib")
        if p:
            files.add(p)


def add_dir(d: Path, files: set):
    if d.is_dir():
        for f in d.rglob("*"):
            if f.is_file():
                files.add(f)


# ── copy helpers ──────────────────────────────────────────────────────────────

TEX_LIKE = {".tex", ".tikz"}


def copy_files(files: set[Path], dest_root: Path, clean: bool):
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True)

    label = "(clean)" if clean else "(with changes)"
    print(f"\nCopying {len(files)} files → {dest_root}  {label}\n")
    for f in sorted(files):
        rel  = f.relative_to(SOURCE)
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if clean and f.suffix in TEX_LIKE:
            text = f.read_text(errors="replace")
            dest.write_text(strip_changes(text), encoding="utf-8")
        else:
            shutil.copy2(f, dest)
        print(f"  {rel}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    files:   set[Path] = set()
    visited: set[Path] = set()

    collect(SOURCE / "paper.tex", visited, files)

    for name in ["sn-jnl.cls", "sn-mathphys-num.bst", "paper.bbl"]:
        p = SOURCE / name
        if p.exists():
            files.add(p)
        else:
            print(f"  [WARN] support file not found: {name}")

    for d in sorted(SOURCE.iterdir()):
        if d.is_dir() and d.name.startswith("graphData"):
            add_dir(d, files)

    pics = SOURCE / "pictures"
    if pics.is_dir():
        for item in sorted(pics.iterdir()):
            if item.is_dir():
                add_dir(item, files)
            else:
                files.add(item)

    # ── version with changes visible ─────────────────────────────────────────
    copy_files(files, OUT_DIR, clean=False)
    zip1 = OUT_DIR.parent / "JMIV_zipped"
    shutil.make_archive(str(zip1), "zip", OUT_DIR)
    print(f"\n✓  {zip1}.zip  ({zip1.with_suffix('.zip').stat().st_size/1e6:.1f} MB)")

    # ── clean version (changes stripped) ─────────────────────────────────────
    copy_files(files, OUT_CLEAN, clean=True)
    zip2 = OUT_CLEAN.parent / "JMIV_zipped_clean"
    shutil.make_archive(str(zip2), "zip", OUT_CLEAN)
    print(f"✓  {zip2}.zip  ({zip2.with_suffix('.zip').stat().st_size/1e6:.1f} MB)\n")


if __name__ == "__main__":
    main()

"""
Collect all files needed to compile paper.tex and package them in
  ../out/JMIV_zipped/
followed by a ../out/JMIV_zipped.zip archive.

.tex / .tikz files are copied AS-IS (they already contain the \\replaced /
\\added / \\deleted markup from the LaTeX `changes` package).
"""

import re
import shutil
from pathlib import Path

SOURCE  = Path("/home/romain22222/visibilityLattices/JMIV2026_rev1")
OUT_DIR = Path("/home/romain22222/visibilityLattices/out/JMIV_zipped")

# Extensions tried (in order) when \includegraphics has no explicit extension
IMAGE_EXTS = [".pdf", ".png", ".jpg", ".jpeg", ".eps"]

# ── helpers ──────────────────────────────────────────────────────────────────

def resolve(ref: str, exts=None) -> Path | None:
    p = SOURCE / ref
    if p.exists():
        return p
    for ext in (exts or []):
        c = SOURCE / (ref + ext)
        if c.exists():
            return c
    return None


def collect(tex_path: Path, visited: set, files: set):
    """Recursively collect all files referenced from a .tex / .tikz file."""
    if tex_path in visited:
        return
    visited.add(tex_path)

    if not tex_path.exists():
        with_tex = tex_path.with_suffix(".tex")
        if with_tex.exists():
            tex_path = with_tex
        else:
            print(f"  [WARN] not found: {tex_path}")
            return

    files.add(tex_path)
    text = tex_path.read_text(errors="replace")

    # \input{...} and \include{...}
    for m in re.finditer(r"\\(?:input|include)\s*\{([^}]+)\}", text):
        ref = m.group(1).strip()
        cand = SOURCE / ref
        if not cand.suffix:
            cand = cand.with_suffix(".tex")
        collect(cand, visited, files)

    # \includegraphics[...]{...}   (skip macro arguments like {#1})
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}", text):
        ref = m.group(1).strip()
        if "#" in ref:
            continue
        p = resolve(ref, IMAGE_EXTS)
        if p:
            files.add(p)
        else:
            print(f"  [WARN] image not found: {ref}")

    # pgfplots data files:   ] {./graphDataXxx/file.txt}
    for m in re.finditer(r"\]\s*\{([^}]+\.txt)\}", text):
        ref = m.group(1).strip()
        p = resolve(ref)
        if p:
            files.add(p)
        else:
            print(f"  [WARN] data file not found: {ref}")

    # \bibliography{name}
    for m in re.finditer(r"\\bibliography\s*\{([^}]+)\}", text):
        p = resolve(m.group(1).strip() + ".bib")
        if p:
            files.add(p)


def add_dir(d: Path, files: set):
    if d.is_dir():
        for f in d.rglob("*"):
            if f.is_file():
                files.add(f)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    files:   set[Path] = set()
    visited: set[Path] = set()

    collect(SOURCE / "paper.tex", visited, files)

    # Always-needed support files
    for name in ["sn-jnl.cls", "sn-mathphys-num.bst", "paper.bbl"]:
        p = SOURCE / name
        if p.exists():
            files.add(p)
        else:
            print(f"  [WARN] support file not found: {name}")

    # All graphData* directories (some paths go through LaTeX macros)
    for d in sorted(SOURCE.iterdir()):
        if d.is_dir() and d.name.startswith("graphData"):
            add_dir(d, files)

    # All pictures/* (sub-dirs accessed via \newcommand with #1 arguments)
    pics = SOURCE / "pictures"
    if pics.is_dir():
        for item in sorted(pics.iterdir()):
            if item.is_dir():
                add_dir(item, files)
            else:
                files.add(item)

    # ── copy ─────────────────────────────────────────────────────────────────
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    print(f"\nCopying {len(files)} files → {OUT_DIR}\n")
    for f in sorted(files):
        rel  = f.relative_to(SOURCE)
        dest = OUT_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        print(f"  {rel}")

    # ── zip ───────────────────────────────────────────────────────────────────
    zip_target = OUT_DIR.parent / "JMIV_zipped"
    shutil.make_archive(str(zip_target), "zip", OUT_DIR)
    size_mb = zip_target.with_suffix(".zip").stat().st_size / 1e6
    print(f"\n✓  {zip_target}.zip  ({size_mb:.1f} MB)\n")


if __name__ == "__main__":
    main()






