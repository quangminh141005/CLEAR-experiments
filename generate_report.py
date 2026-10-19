#!/usr/bin/env python3
"""Generate a Vietnamese Markdown/Word report after server-side experiments."""
import argparse
from pathlib import Path

from evaluation.reporting import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], metavar="DATASET=RUN_DIRECTORY")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--template-docx", help="Optional previous CXR-LLaVA DOCX style template")
    parser.add_argument("--overwrite", action="store_true", help="Replace only this generator's two named reports")
    args = parser.parse_args()
    paths = {}
    for item in args.run:
        name, separator, path = item.partition("=")
        if not separator or name not in {"indiana", "padchest", "chexpert"} or name in paths:
            parser.error("Each --run must be a unique indiana|padchest|chexpert=RUN_DIRECTORY")
        paths[name] = path
    for path in generate(paths, args.output_dir, template=args.template_docx, overwrite=args.overwrite):
        print(path.resolve())


if __name__ == "__main__":
    main()
