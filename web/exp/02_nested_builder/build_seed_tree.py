#!/usr/bin/env python3
"""Construct and persist a seed website tree for downstream experiments.

The generated artefacts mirror Stage 2 outputs but are stored under
``web/exp/02_nested_builder/tree/`` so that experiment runs can later extract
deterministic subtrees without rerunning the full Stage 01–02 pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .build_tree_design import (
        build_tree_nodes,
        make_site_manifest,
        write_site_manifest,
        write_tree_bundle,
    )
except ImportError:
    # Allow running this script directly without package context
    if __package__ in (None, ""):
        sys.path.append(str(Path(__file__).resolve().parent))
        from build_tree_design import (  # type: ignore
            build_tree_nodes,
            make_site_manifest,
            write_site_manifest,
            write_tree_bundle,
        )
    else:
        raise


TREE_BASE = Path(__file__).resolve().parent / "tree"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree-id",
        type=str,
        help=(
            "Directory name for the seed tree under web/exp/02_nested_builder/tree/. "
            "Defaults to d{depth}_w{width}_s{seed}."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional explicit output directory (overrides --tree-id).",
    )
    parser.add_argument("--depth", type=int, default=3, help="Tree depth (>=1).")
    parser.add_argument("--width", type=int, default=2, help="Branching factor (>=1).")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic generation.")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model forwarded to Stage 01 builders.",
    )
    parser.add_argument("--max-tokens", type=int, default=768, help="Maximum tokens for Stage 01 generation.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional nucleus sampling parameter.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing contents in the output directory if present.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display even when stderr is a TTY.",
    )
    return parser.parse_args()


def _prepare_output_root(args: argparse.Namespace) -> tuple[Path, str]:
    if args.output_root:
        output_root = args.output_root
        tree_id = args.tree_id or output_root.name
    else:
        TREE_BASE.mkdir(parents=True, exist_ok=True)
        tree_id = args.tree_id or f"d{args.depth}_w{args.width}_s{args.seed}"
        output_root = TREE_BASE / tree_id

    if output_root.exists() and not args.force:
        existing_contents = list(output_root.iterdir()) if output_root.is_dir() else []
        if existing_contents:
            raise SystemExit(
                f"Output directory {output_root} already exists with contents. "
                "Re-run with --force to overwrite."
            )

    output_root.mkdir(parents=True, exist_ok=True)
    return output_root, tree_id


def main() -> None:
    args = parse_args()
    output_root, tree_id = _prepare_output_root(args)

    show_progress = not args.no_progress and sys.stderr.isatty()
    nodes = build_tree_nodes(
        args.depth,
        args.width,
        args.seed,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        progress=show_progress,
    )

    write_tree_bundle(output_root, nodes)

    manifest = make_site_manifest(
        depth=args.depth,
        width=args.width,
        base_seed=args.seed,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_root=output_root,
        num_pages=len(nodes),
        extra={
            "kind": "seed_tree",
            "tree_id": tree_id,
        },
    )
    manifest_path = write_site_manifest(output_root, manifest)

    print(f"Seed tree written to {output_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
