"""Command-line phase gates for planning, training, flow work, and upload."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from grokking_lab.config import load_config
from grokking_lab.definitions import summarize_raw_run
from grokking_lab.flow import extract_run
from grokking_lab.posthoc import extend_run
from grokking_lab.train import train_sweep, write_behavior_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grokking-lab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="print scale and artifact estimates")
    plan.add_argument("--config", type=Path, required=True)

    train = commands.add_parser("train", help="run behavior-only training")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="auto")
    train.add_argument(
        "--confirm-scale",
        action="store_true",
        help="required above 10,000 aggregate optimizer steps",
    )

    extract = commands.add_parser("extract-flows", help="write raw flow values")
    extract.add_argument("--run", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--workers", type=int, default=1)
    extract.add_argument("--resume", action="store_true")
    extract.add_argument("--compression-level", type=int, default=6)
    extract.add_argument("--acknowledge-phase-gate", action="store_true", required=True)

    summarize = commands.add_parser("summarize", help="compute numbered definitions")
    summarize.add_argument("--raw-run", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--acknowledge-phase-gate", action="store_true", required=True)

    behavior = commands.add_parser("report-behavior", help="aggregate completed behavior-only runs")
    behavior.add_argument("--runs", type=Path, required=True)

    posthoc = commands.add_parser(
        "posthoc-extend", help="resume one run in a separate post-hoc artifact"
    )
    posthoc.add_argument("--source-run", type=Path, required=True)
    posthoc.add_argument("--output", type=Path, required=True)
    posthoc.add_argument("--target-epoch", type=int, required=True)
    posthoc.add_argument("--checkpoint-every", type=int, default=100)
    posthoc.add_argument("--device", default="auto")
    posthoc.add_argument("--acknowledge-posthoc", action="store_true", required=True)

    upload = commands.add_parser("upload", help="upload an artifact folder to HF")
    upload.add_argument("--source", type=Path, required=True)
    upload.add_argument("--repo-id", required=True)
    upload.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
    upload.add_argument("--commit-message", default="Upload grokking experiment artifacts")
    upload.add_argument("--confirm-upload", action="store_true", required=True)
    return parser


def _upload(args: argparse.Namespace) -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is not set; request credentials before upload")
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("install the 'hf' optional dependency") from error
    api = HfApi()
    api.create_repo(args.repo_id, repo_type=args.repo_type, exist_ok=True, private=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=args.source,
        commit_message=args.commit_message,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        print(json.dumps(load_config(args.config).plan(), indent=2, sort_keys=True))
    elif args.command == "train":
        config = load_config(args.config)
        plan = config.plan()
        if plan["optimizer_steps"] > 10_000 and not args.confirm_scale:
            raise RuntimeError(
                "large training is phase-gated; obtain scale confirmation, "
                "then pass --confirm-scale"
            )
        train_sweep(config, args.output, args.device)
    elif args.command == "extract-flows":
        extract_run(
            args.run,
            args.output,
            args.device,
            args.workers,
            args.resume,
            args.compression_level,
        )
    elif args.command == "summarize":
        summarize_raw_run(args.raw_run, args.output)
    elif args.command == "report-behavior":
        print(write_behavior_report(args.runs))
    elif args.command == "posthoc-extend":
        print(
            extend_run(
                args.source_run,
                args.output,
                args.target_epoch,
                args.checkpoint_every,
                args.device,
            )
        )
    else:
        _upload(args)


if __name__ == "__main__":
    main()
