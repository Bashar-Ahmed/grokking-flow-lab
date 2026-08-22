#!/usr/bin/env python3
"""Stream and validate a complete raw-flow artifact tree."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


def audit_run(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    files = json.loads((run_dir / "files.json").read_text())
    run_id = str(manifest["run_id"])
    kinds = tuple(manifest["flow_kinds"])
    split_of = {int(row["index"]): str(row["split"]) for row in manifest["selected_examples"]}
    if Counter(split_of.values()) != {"train": 8, "test": 8}:
        raise ValueError(f"unexpected selected-example splits in {run_id}")
    expected_pairs = {(index, kind) for index in split_of for kind in kinds}
    expected_records = len(expected_pairs)
    scale = float(manifest["flow_encoding"]["scale"])
    num_nodes = len(manifest["node_labels"])
    totals: Counter[str] = Counter()

    for file_row in files:
        epoch = int(file_row["epoch"])
        path = run_dir / file_row["path"]
        pairs: set[tuple[int, str]] = set()
        checkpoint_splits: Counter[str] = Counter()
        checkpoint_kinds: Counter[str] = Counter()
        records = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                records += 1
                if record["schema_version"] != 2:
                    raise ValueError(f"schema mismatch in {path}")
                if record["run_id"] != run_id or int(record["epoch"]) != epoch:
                    raise ValueError(f"provenance mismatch in {path}")
                example_index = int(record["example_index"])
                split = str(record["split"])
                kind = str(record["flow_kind"])
                if split_of.get(example_index) != split:
                    raise ValueError(f"split mismatch in {path}")
                if kind not in kinds or (example_index, kind) in pairs:
                    raise ValueError(f"flow-kind/example mismatch in {path}")
                if float(record["flow_scale"]) != scale:
                    raise ValueError(f"flow-scale mismatch in {path}")
                pairs.add((example_index, kind))
                checkpoint_splits[split] += 1
                checkpoint_kinds[kind] += 1
                totals[f"split:{split}"] += 1
                totals[f"kind:{kind}"] += 1
                totals[f"status:{record['status']}"] += 1

                if record["status"] != "ok":
                    continue
                path_sum = sum(float(row[1]) for row in record["canonical_paths"])
                if abs(path_sum - scale) > 0.01:
                    raise ValueError(f"canonical paths are not normalized in {path}")
                incoming = [0.0] * num_nodes
                outgoing = [0.0] * num_nodes
                for tail, head, value in record["edges"]:
                    tail = int(tail)
                    head = int(head)
                    value = float(value)
                    if not (0 <= tail < head < num_nodes) or value < 0:
                        raise ValueError(f"invalid edge in {path}")
                    outgoing[tail] += value
                    incoming[head] += value
                if abs(outgoing[0] - scale) > 0.01 or abs(incoming[-1] - scale) > 0.01:
                    raise ValueError(f"source/sink normalization mismatch in {path}")
                if (
                    max(
                        (abs(incoming[node] - outgoing[node]) for node in range(1, num_nodes - 1)),
                        default=0.0,
                    )
                    > 0.01
                ):
                    raise ValueError(f"flow conservation mismatch in {path}")

        if records != expected_records or int(file_row["records"]) != expected_records:
            raise ValueError(f"record-count mismatch in {path}")
        if pairs != expected_pairs:
            raise ValueError(f"missing example/flow-kind pair in {path}")
        if checkpoint_splits != {"train": expected_records // 2, "test": expected_records // 2}:
            raise ValueError(f"checkpoint split imbalance in {path}")
        if any(checkpoint_kinds[kind] != len(split_of) for kind in kinds):
            raise ValueError(f"checkpoint flow-kind imbalance in {path}")

    return {
        "run_id": run_id,
        "checkpoints": len(files),
        "records": len(files) * expected_records,
        "compressed_bytes": sum(int(row["bytes"]) for row in files),
        "counts": dict(totals),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dirs = sorted(path for path in args.root.iterdir() if path.is_dir())
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        runs = list(executor.map(audit_run, run_dirs))
    counts: Counter[str] = Counter()
    for run in runs:
        counts.update(run["counts"])
    result = {
        "schema_version": 1,
        "status": "passed",
        "runs": len(runs),
        "checkpoints": sum(run["checkpoints"] for run in runs),
        "records": sum(run["records"] for run in runs),
        "compressed_bytes": sum(run["compressed_bytes"] for run in runs),
        "counts": dict(sorted(counts.items())),
        "run_results": runs,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
