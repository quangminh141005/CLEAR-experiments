#!/usr/bin/env python3
"""Explicit server-side asset acquisition/registration; never called by evaluation."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.inference import ASSET_FILES, observation_list
from evaluation.storage import file_hash, stable_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="assets/clear")
    parser.add_argument("--revision", required=True, help="Official HF commit SHA, or branch resolved and recorded once")
    parser.add_argument("--inference-mode", choices=["clear", "direct"], default="clear")
    parser.add_argument("--register-local", action="store_true", help="Only hash existing files; no network")
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "assets.json").exists():
        raise FileExistsError("assets.json exists; choose another directory to register a new snapshot")
    revision = args.revision
    names = ASSET_FILES if args.inference_mode == "clear" else ASSET_FILES[:1]
    if args.register_local and (len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower())):
        parser.error("Local registration requires the known 40-character official snapshot commit")
    if not args.register_local:
        from huggingface_hub import HfApi, hf_hub_download
        revision = HfApi().model_info("peterhan91/CLEAR", revision=revision).sha
        for name in names:
            source = Path(hf_hub_download("peterhan91/CLEAR", filename=name, revision=revision))
            target = root / name
            if target.exists():
                if file_hash(target) != file_hash(source):
                    raise FileExistsError(f"Existing asset differs: {target}")
                continue
            # Link the HF cache on the same filesystem; avoid duplicating multi-GB assets.
            try:
                target.symlink_to(source.resolve())
            except OSError:
                shutil.copyfile(source, target)
    hashes = {name: file_hash(root / name) for name in names}
    concepts = observation_list(root / "mimic_concepts.csv") if args.inference_mode == "clear" else []
    if args.inference_mode == "clear" and len(concepts) != 368294:
        raise ValueError(f"Official full bank requires 368294 observations; found {len(concepts)}")
    write_json(root / "assets.json", {"repository": "peterhan91/CLEAR", "revision": revision,
        "sha256": hashes, "expected_concepts": len(concepts), "observation_order_sha256": stable_hash(concepts) if concepts else None,
        "registration": "local user-supplied provenance" if args.register_local else "resolved official HF snapshot",
        "alignment_note": "Embedding row i is concept_idx i; source files must come from the same official revision"})
    print(f"Registered {len(concepts)} observations at {root.resolve()}, revision {revision}")


if __name__ == "__main__":
    main()
