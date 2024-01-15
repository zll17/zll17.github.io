#!/usr/bin/env python3
"""
Kramdown only treats $$...$$ as inline/block math, not $...$ (Typora/Hexo style).
Convert $...$ -> $$...$$ outside of $$...$$ display regions.
Skips {% ... %} liquid blocks and escaped \\$.
"""
from __future__ import annotations

import pathlib
import sys


def upgrade_single_dollar_math(segment: str) -> str:
    out: list[str] = []
    i = 0
    n = len(segment)
    while i < n:
        if segment.startswith("{%", i):
            j = segment.find("%}", i)
            if j == -1:
                out.append(segment[i:])
                break
            j += 2
            out.append(segment[i:j])
            i = j
            continue
        if i < n - 1 and segment[i] == "\\" and segment[i + 1] == "$":
            out.append("\\$")
            i += 2
            continue
        if segment[i] == "$":
            if i < n - 1 and segment[i + 1] == "$":
                out.append("$$")
                i += 2
                continue
            j = i + 1
            while j < n:
                if segment[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if segment[j] == "$":
                    break
                j += 1
            if j >= n:
                out.append(segment[i:])
                break
            inner = segment[i + 1 : j]
            out.append("$$" + inner + "$$")
            i = j + 1
            continue
        out.append(segment[i])
        i += 1
    return "".join(out)


def process_markdown_body(body: str) -> str:
    parts = body.split("$$")
    new_parts: list[str] = []
    for idx, p in enumerate(parts):
        if idx % 2 == 0:
            new_parts.append(upgrade_single_dollar_math(p))
        else:
            new_parts.append(p)
    return "$$".join(new_parts)


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise SystemExit("expected YAML front matter")
    fm_end = text.index("---", 3) + 3
    front = text[:fm_end]
    body = text[fm_end:]
    path.write_text(front + process_markdown_body(body), encoding="utf-8")
    print(f"updated: {path}")


if __name__ == "__main__":
    main()
