"""Check inline relative Markdown links against files in the checkout.

External URLs and heading anchors are outside this check's scope.
"""

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)")


def missing_links(path):
    fence = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            value = marker.group(1)
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = None
            continue
        if fence:
            continue
        for match in LINK.finditer(line):
            target = match.group(1).strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            destination = path.parent / unquote(url.path)
            if not destination.exists():
                yield number, target


def main():
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "*.md"], cwd=ROOT,
    ).decode().split("\0")
    failures = 0
    for name in filter(None, files):
        for number, target in missing_links(ROOT / name):
            print(f"{name}:{number}: missing link target: {target}")
            failures += 1
    if failures:
        return 1
    print("Relative documentation link targets exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
