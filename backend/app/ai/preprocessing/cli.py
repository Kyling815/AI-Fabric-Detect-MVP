"""Command-line interface for safe preprocessing-v2 batch runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from backend.app.ai.preprocessing.config import PreprocessConfig
from backend.app.ai.preprocessing.pipeline import process_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess already-captured fabric images with v2.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--fabric-type")
    parser.add_argument("--split", choices=("train", "validation", "test", "unspecified"), default="unspecified")
    parser.add_argument("--save-texture-view", action="store_true")
    parser.add_argument("--extract-patches", action="store_true")
    parser.add_argument("--patch-sizes", nargs="+", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = PreprocessConfig.load(args.config) if args.config else PreprocessConfig()
        if args.patch_sizes:
            config = config.with_overrides({"patch_sizes": tuple(args.patch_sizes)})
        summary = process_directory(
            input_root=args.input,
            output_root=args.output,
            config=config,
            registry_path=args.registry,
            metadata_path=args.metadata,
            fabric_type=args.fabric_type,
            split=args.split,
            extract_patches=args.extract_patches,
            save_texture_view=args.save_texture_view,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "fatal_error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps({"status": "dry_run" if args.dry_run else "complete", **summary.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
