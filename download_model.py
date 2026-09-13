#!/usr/bin/env python3
"""Download one immutable Hugging Face snapshot for later offline benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DEFAULT_REPO = "CMKL/MANGO1.5-Qwen3.5-9B"
DEFAULT_DESTINATION = Path("/data/users/g6914500746/models/MANGO1.5-Qwen3.5-9B")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--revision", default="main",
                        help="Branch, tag, or commit. It is resolved and recorded as a commit SHA.")
    args = parser.parse_args()

    args.destination.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.model_info(args.repo_id, revision=args.revision, files_metadata=True)
    commit_sha = info.sha
    expected_bytes = sum(sibling.size or 0 for sibling in info.siblings)
    print(f"Downloading {args.repo_id}@{commit_sha}")
    print(f"Expected repository size: {expected_bytes / 2**30:.2f} GiB")
    local_path = snapshot_download(
        repo_id=args.repo_id,
        revision=commit_sha,
        local_dir=args.destination,
    )
    manifest = {
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "commit_sha": commit_sha,
        "expected_bytes": expected_bytes,
        "local_path": str(Path(local_path).resolve()),
    }
    with (args.destination / "local_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
