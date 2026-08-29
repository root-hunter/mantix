#!/usr/bin/env python3
"""Compare the median latency in two Mantix benchmark CSV snapshots."""

import argparse
import csv
import sys
from pathlib import Path


def load(path: Path, library: str) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        required = {"library", "benchmark", "precision_bits", "median_ns_per_op"}
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise ValueError(f"{path}: unsupported benchmark CSV")
        return {
            (row["benchmark"], int(row["precision_bits"])): row
            for row in rows
            if row["library"] == library
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--library", default="mantix")
    args = parser.parse_args()

    try:
        baseline = load(args.baseline, args.library)
        candidate = load(args.candidate, args.library)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    common = sorted(baseline.keys() & candidate.keys())
    if not common:
        print(f"no common {args.library!r} benchmarks", file=sys.stderr)
        return 1

    comparable_fields = (
        "compiler", "architecture", "mantix_backend", "iterations", "samples"
    )
    warned = False
    for key in common:
        old_row = baseline[key]
        new_row = candidate[key]
        differences = [
            field for field in comparable_fields
            if old_row.get(field) != new_row.get(field)
        ]
        if differences:
            print(
                "warning: benchmark configurations differ for "
                f"{key[0]}/{key[1]} bits: {', '.join(differences)}",
                file=sys.stderr,
            )
            warned = True
    if warned:
        print("warning: treat these deltas with caution", file=sys.stderr)

    old_versions = sorted({row["version"] for row in baseline.values()})
    new_versions = sorted({row["version"] for row in candidate.values()})
    print(f"{args.library}: {', '.join(old_versions)} -> {', '.join(new_versions)}")

    print(f"{'benchmark':<20} {'bits':>6} {'baseline ns':>13} "
          f"{'candidate ns':>13} {'delta':>9} {'speedup':>9}")
    for name, precision in common:
        old = float(baseline[(name, precision)]["median_ns_per_op"])
        new = float(candidate[(name, precision)]["median_ns_per_op"])
        delta = (new / old - 1.0) * 100.0
        speedup = old / new
        print(f"{name:<20} {precision:>6} {old:>13.3f} {new:>13.3f} "
              f"{delta:>+8.2f}% {speedup:>8.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
