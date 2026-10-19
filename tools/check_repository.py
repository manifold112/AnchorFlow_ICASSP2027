"""Check local documentation links, SVG structure, and manuscript table schemas."""

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    checked = 0
    for path in ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        targets = re.findall(r"\]\(([^)]+)\)", text) + re.findall(r'src="([^"]+)"', text)
        for target in targets:
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not (path.parent / target).exists():
                raise FileNotFoundError(f"{path.relative_to(ROOT)} -> {target}")
            checked += 1
    for path in (ROOT / "docs" / "assets").glob("*.svg"):
        root = ET.parse(path).getroot()
        if root.tag != "{http://www.w3.org/2000/svg}svg":
            raise ValueError(f"invalid SVG root: {path}")
    with (ROOT / "benchmarks" / "paper_results.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 8:
        raise ValueError("Table 1 requires two protocols and four backbones")
    for row in rows:
        for key, value in row.items():
            if key not in ("protocol", "backbone") and not 0 <= float(value) <= 100:
                raise ValueError(f"invalid manuscript statistic: {key}")
    print(f"Checked {checked} local links, SVG structure and manuscript Table 1 schema")


if __name__ == "__main__":
    main()
