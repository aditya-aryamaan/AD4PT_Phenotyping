"""Compare manually verified trial counts with run JSONs; plot report figures."""

import argparse
import csv
import json
from pathlib import Path


def generate(trials: Path, output: Path) -> int:
    rows = []
    seen = set()
    with trials.open(newline="") as handle:
        for trial in csv.DictReader(handle):
            identity = trial["trial_id"].strip()
            if not identity or identity in seen:
                raise ValueError("trial_id must be nonempty and unique")
            seen.add(identity)
            path = Path(trial["counts_json"])
            run = json.loads(path.read_text())
            if run["stop_reason"] in ("error", "no_stream"):
                raise ValueError(f"{identity}: incomplete run ({run['stop_reason']})")
            row = {**trial, "stop_reason": run["stop_reason"]}
            for label in ("ripe", "unripe"):
                actual = int(trial[f"actual_{label}"])
                predicted = run[label]
                if actual < 0 or type(predicted) is not int or predicted < 0:
                    raise ValueError(f"{identity}: counts must be nonnegative integers")
                row[f"predicted_{label}"] = predicted
                row[f"error_{label}"] = predicted - actual
                row[f"absolute_error_{label}"] = abs(predicted - actual)
            row["actual_total"] = int(trial["actual_ripe"]) + int(
                trial["actual_unripe"]
            )
            row["predicted_total"] = run["ripe"] + run["unripe"]
            row["total_error"] = row["predicted_total"] - row["actual_total"]
            rows.append(row)
    if not rows:
        raise ValueError(
            "No trials yet. Add completed runs and manual counts to the CSV."
        )
    # Import plotting only for reports, never in the live detector.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    with (output / "count-comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "trials": len(rows),
        "ripe_mean_absolute_count_error": sum(r["absolute_error_ripe"] for r in rows)
        / len(rows),
        "unripe_mean_absolute_count_error": sum(
            r["absolute_error_unripe"] for r in rows
        )
        / len(rows),
        "total_mean_absolute_count_error": sum(abs(r["total_error"]) for r in rows)
        / len(rows),
        "note": (
            "Count agreement only; these metrics do not measure "
            "detection precision or recall."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = [
        "# Counting results",
        "",
        "Signed error = system count minus manual count. "
        "These are counting measurements, not detection precision/recall.",
        "",
        "| Trial | Class | Manual | System | Signed error | Absolute error (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        for label in ("ripe", "unripe", "total"):
            actual = int(row[f"actual_{label}"])
            predicted = int(row[f"predicted_{label}"])
            error = predicted - actual
            percent = f"{100 * abs(error) / actual:.1f}" if actual else "N/A"
            identity = str(row["trial_id"]).replace("|", "/")
            table.append(
                f"| {identity} | {label.capitalize()} | {actual} | "
                f"{predicted} | {error:+d} | {percent} |"
            )
    (output / "results-table.md").write_text("\n".join(table) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(rows) * 1.2), 4.5))
    for axis, label in zip(axes, ("ripe", "unripe"), strict=True):
        positions = list(range(len(rows)))
        axis.bar(
            [p - 0.2 for p in positions],
            [int(r[f"actual_{label}"]) for r in rows],
            0.4,
            label="Manual count",
        )
        axis.bar(
            [p + 0.2 for p in positions],
            [r[f"predicted_{label}"] for r in rows],
            0.4,
            label="System count",
        )
        axis.set_xticks(
            positions, [r["trial_id"] for r in rows], rotation=45, ha="right"
        )
        axis.set(title=label.capitalize(), ylabel="Strawberries", xlabel="Trial")
        maximum = max(
            max(int(r[f"actual_{label}"]) for r in rows),
            max(r[f"predicted_{label}"] for r in rows),
        )
        axis.set_ylim(0, max(1, maximum * 1.3))
        axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"manual-vs-system.{suffix}", dpi=200)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(max(7, len(rows) * 0.6), 4))
    axis.axhline(0, color="black", linewidth=0.8)
    for label, offset in (("ripe", -0.2), ("unripe", 0.2)):
        axis.bar(
            [i + offset for i in range(len(rows))],
            [r[f"error_{label}"] for r in rows],
            width=0.4,
            label=label.capitalize(),
        )
    axis.set_xticks(
        range(len(rows)), [r["trial_id"] for r in rows], rotation=45, ha="right"
    )
    axis.set(
        xlabel="Trial",
        ylabel="System count − manual count",
        title="Counting error by trial",
    )
    axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"count-error.{suffix}", dpi=200)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, default=Path("reports/testing/results"))
    args = parser.parse_args()
    try:
        raise SystemExit(generate(args.trials, args.output))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Report error: {exc}\n")
