"""Apply individually reviewed alignment fixes, retaining original label backups."""

import argparse
import hashlib
import json
from pathlib import Path


def transform_row(row, transform):
    cls, x, y, w, h = row
    if transform == "180":
        return [cls, 1 - x, 1 - y, w, h]
    if transform == "ccw90":
        return [cls, y, 1 - x, h, w]
    if transform == "identity":
        return row
    raise ValueError(f"Unsupported reviewed transform: {transform}")


def apply(root, corrections):
    changes = json.loads(corrections.read_text())
    for record in changes:
        relative = (
            Path(record["split"]) / Path(record["image"]).with_suffix(".txt").name
        )
        label = root / "labels" / relative
        backup = root / "labels-before-alignment" / relative
        before = backup.read_text() if backup.exists() else label.read_text()
        if hashlib.sha256(before.encode()).hexdigest() != record["before_sha256"]:
            raise ValueError(f"Labels differ from the reviewed source: {label}")
        rows = [
            transform_row(list(map(float, line.split())), record["transform"])
            for line in before.splitlines()
        ]
        rows += record.get("additions", [])
        after = "".join(
            f"{int(row[0])} " + " ".join(repr(v) for v in row[1:]) + "\n"
            for row in rows
        )
        if label.read_text() not in (before, after):
            raise ValueError(f"Refusing to overwrite other edits: {label}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_text(before)
        label.write_text(after)
    (root / "alignment-corrections.json").write_text(
        json.dumps(changes, indent=2) + "\n"
    )
    print(f"Applied {len(changes)} reviewed image-label corrections; backups retained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("datasets/strawberry-adaptation")
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    apply(args.root, args.corrections)
