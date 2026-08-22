"""Numbered candidate formulas computed only from stored raw flow records."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import fmean
from typing import Any

from grokking_lab.io import write_csv


@dataclass(frozen=True)
class Graph:
    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    values: tuple[float, ...]
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.values):
            raise ValueError("one value is required per edge")
        if any(value < -self.tolerance for value in self.values):
            raise ValueError("flow values must be nonnegative")
        if any(tail >= head for tail, head in self.edges):
            raise ValueError("node indices must be a strict topological order")
        incoming, outgoing = self.node_values()
        error = max(
            (abs(incoming[node] - outgoing[node]) for node in range(1, self.num_nodes - 1)),
            default=0.0,
        )
        if error > 1e-7:
            raise ValueError(f"flow conservation error is {error}")

    @property
    def value(self) -> float:
        return sum(
            value for (tail, _), value in zip(self.edges, self.values, strict=True) if tail == 0
        )

    def adjacency(self) -> tuple[list[list[int]], list[list[int]]]:
        incoming = [[] for _ in range(self.num_nodes)]
        outgoing = [[] for _ in range(self.num_nodes)]
        for edge_index, (tail, head) in enumerate(self.edges):
            outgoing[tail].append(edge_index)
            incoming[head].append(edge_index)
        return incoming, outgoing

    def node_values(self) -> tuple[list[float], list[float]]:
        incoming = [0.0] * self.num_nodes
        outgoing = [0.0] * self.num_nodes
        for (tail, head), value in zip(self.edges, self.values, strict=True):
            outgoing[tail] += max(0.0, value)
            incoming[head] += max(0.0, value)
        return incoming, outgoing


def graph_from_record(record: dict[str, Any], num_nodes: int) -> Graph:
    edges = tuple((int(row[0]), int(row[1])) for row in record["edges"])
    scale = float(record.get("flow_scale", 1.0))
    if scale <= 0:
        raise ValueError("flow_scale must be positive")
    values = tuple(float(row[2]) / scale for row in record["edges"])
    return Graph(num_nodes, edges, values)


def path_excess(graph: Graph, path: tuple[int, ...]) -> float:
    if not path:
        raise ValueError("path cannot be empty")
    _, outgoing = graph.node_values()
    excess = graph.values[path[0]]
    for edge_index in path[1:]:
        tail, _ = graph.edges[edge_index]
        excess -= outgoing[tail] - graph.values[edge_index]
    return excess


def safe_paths(graph: Graph, min_length: int = 2) -> list[tuple[int, ...]]:
    _, outgoing_edges = graph.adjacency()
    _, outgoing_values = graph.node_values()
    result: list[tuple[int, ...]] = []

    def extend(path: tuple[int, ...], excess: float) -> None:
        if len(path) >= min_length:
            result.append(path)
        head = graph.edges[path[-1]][1]
        for next_edge in outgoing_edges[head]:
            next_excess = excess - (outgoing_values[head] - graph.values[next_edge])
            if next_excess > graph.tolerance:
                extend((*path, next_edge), next_excess)

    for edge_index, value in enumerate(graph.values):
        if value > graph.tolerance:
            extend((edge_index,), value)
    return sorted(result, key=lambda path: (path[0], len(path), path))


def _eligible_nodes(graph: Graph) -> set[int]:
    incoming_edges, outgoing_edges = graph.adjacency()
    incoming, _ = graph.node_values()
    return {
        node
        for node in range(1, graph.num_nodes - 1)
        if len(incoming_edges[node]) >= 2
        and len(outgoing_edges[node]) >= 2
        and incoming[node] > graph.tolerance
    }


def _eligible_path(graph: Graph, path: tuple[int, ...], nodes: set[int]) -> bool:
    return any(graph.edges[left][1] in nodes for left, _ in pairwise(path))


def definition_01(graph: Graph) -> float:
    incoming_edges, outgoing_edges = graph.adjacency()
    incoming, _ = graph.node_values()
    numerator = 0.0
    denominator = 0.0
    for node in _eligible_nodes(graph):
        denominator += incoming[node]
        for left in incoming_edges[node]:
            for right in outgoing_edges[node]:
                numerator += max(0.0, graph.values[left] + graph.values[right] - incoming[node])
    return numerator / denominator if denominator else 0.0


def _maximal(graph: Graph, paths: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    path_set = set(paths)
    incoming_edges, outgoing_edges = graph.adjacency()
    result = []
    for path in paths:
        start = graph.edges[path[0]][0]
        end = graph.edges[path[-1]][1]
        left = any((edge, *path) in path_set for edge in incoming_edges[start])
        right = any((*path, edge) in path_set for edge in outgoing_edges[end])
        if not left and not right:
            result.append(path)
    return result


def definition_02(graph: Graph) -> float:
    nodes = _eligible_nodes(graph)
    paths = [path for path in safe_paths(graph) if _eligible_path(graph, path, nodes)]
    return sum(path_excess(graph, path) for path in _maximal(graph, paths)) / graph.value


def _descendants(graph: Graph) -> list[list[bool]]:
    reach = [[False] * graph.num_nodes for _ in range(graph.num_nodes)]
    _, outgoing = graph.adjacency()
    for node in range(graph.num_nodes - 1, -1, -1):
        for edge in outgoing[node]:
            head = graph.edges[edge][1]
            reach[node][head] = True
            for descendant, reachable in enumerate(reach[head]):
                reach[node][descendant] |= reachable
    return reach


def _contains(path: tuple[int, ...], subpath: tuple[int, ...]) -> bool:
    width = len(subpath)
    return any(path[start : start + width] == subpath for start in range(len(path) - width + 1))


def _can_cooccur(
    graph: Graph, left: tuple[int, ...], right: tuple[int, ...], reach: list[list[bool]]
) -> bool:
    if _contains(left, right) or _contains(right, left):
        return True
    for leading, trailing in ((left, right), (right, left)):
        for overlap in range(1, min(len(leading), len(trailing))):
            if leading[-overlap:] == trailing[:overlap]:
                return True
        end = graph.edges[leading[-1]][1]
        start = graph.edges[trailing[0]][0]
        if end == start or reach[end][start]:
            return True
    return False


def definition_03(graph: Graph) -> float:
    nodes = _eligible_nodes(graph)
    candidates = [path for path in safe_paths(graph) if _eligible_path(graph, path, nodes)]
    ranked = sorted(
        ((path_excess(graph, path), path) for path in candidates),
        key=lambda item: (-item[0], item[1]),
    )
    reach = _descendants(graph)
    selected: list[tuple[int, ...]] = []
    total = 0.0
    for excess, path in ranked:
        if all(not _can_cooccur(graph, path, other, reach) for other in selected):
            selected.append(path)
            total += excess
    return total / graph.value


def definition_04(graph: Graph) -> float:
    nodes = _eligible_nodes(graph)
    paths = [path for path in safe_paths(graph) if _eligible_path(graph, path, nodes)]
    return (graph.value + sum(path_excess(graph, path) * len(path) for path in paths)) / graph.value


def compute_definitions(graph: Graph) -> dict[str, float]:
    return {
        "definition_01": definition_01(graph),
        "definition_02": definition_02(graph),
        "definition_03": definition_03(graph),
        "definition_04": definition_04(graph),
    }


def summarize_raw_run(raw_run_dir: Path, output_dir: Path) -> Path:
    """Compute numbered outputs from raw files and write an explanatory report."""

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((raw_run_dir / "manifest.json").read_text())
    num_nodes = len(manifest["node_labels"])
    output_path = output_dir / "definitions.jsonl.gz"
    rows: list[dict[str, Any]] = []
    with gzip.open(output_path, "wt", encoding="utf-8") as output_handle:
        for raw_path in sorted(raw_run_dir.glob("epoch_*.jsonl.gz")):
            with gzip.open(raw_path, "rt", encoding="utf-8") as input_handle:
                for line in input_handle:
                    record = json.loads(line)
                    if record["status"] != "ok":
                        continue
                    values = compute_definitions(graph_from_record(record, num_nodes))
                    row = {
                        key: record[key]
                        for key in (
                            "run_id",
                            "epoch",
                            "example_index",
                            "split",
                            "flow_kind",
                        )
                    }
                    row.update(values)
                    output_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                    rows.append(row)

    aggregate = []
    keys = sorted({(int(row["epoch"]), str(row["flow_kind"])) for row in rows})
    for epoch, kind in keys:
        group = [row for row in rows if int(row["epoch"]) == epoch and row["flow_kind"] == kind]
        aggregate.append(
            {
                "epoch": epoch,
                "flow_kind": kind,
                "n": len(group),
                **{
                    name: fmean(float(row[name]) for row in group)
                    for name in (
                        "definition_01",
                        "definition_02",
                        "definition_03",
                        "definition_04",
                    )
                },
            }
        )
    write_csv(output_dir / "aggregate.csv", aggregate)
    (output_dir / "REPORT.md").write_text(_definition_report(raw_run_dir.name, len(rows)))
    return output_dir


def _definition_report(run_id: str, rows: int) -> str:
    return f"""# Numbered-definition report: `{run_id}`

This report covers {rows} nondegenerate raw flow graphs. All quantities are computed
after training from stored edge marginals; none changes the model or checkpoints.

## Definition-01

At each ambiguous internal junction, sum the positive two-edge lower bounds and
divide by total throughput across those junctions.

## Definition-02

Sum the lower-bound values of maximal eligible paths and divide by source flow.
Overlapping paths can both contribute, so this quantity is not asserted to be a
union-mass lower bound.

## Definition-03

Enumerate eligible paths, greedily select paths that cannot coexist on one complete
source-to-sink path, sum their lower bounds, and divide by source flow. This is a
conservative lower bound for the selected union.

## Definition-04

Start with source flow, then add each eligible path's lower bound multiplied by its
edge length, and divide by source flow. Nested and overlapping paths both contribute;
this is an integral, not a union-mass bound.

The names are intentionally numbered. Comparative evidence should determine which,
if any, deserves a semantic label.
"""
