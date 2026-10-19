#!/usr/bin/env python3
"""Encode SFR diagnostic prompts once, separately from CLEAR image inference."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.comparison import paper_labels, prompt_texts
from evaluation.inference import validate_matrix
from evaluation.storage import stable_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["indiana", "padchest", "chexpert"])
    parser.add_argument("--cohort", help="cohort.json from evaluate.py --prepare-only; supports --all-findings/custom labels")
    parser.add_argument("--model", default="Salesforce/SFR-Embedding-Mistral", help="HF ID or local snapshot directory")
    parser.add_argument("--revision", required=True, help="SFR immutable commit SHA (also record it for local models)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--format", choices=["raw", "instruction"], default="raw")
    parser.add_argument("--instruction", default="Given medical radiology concepts, retrieve relevant passages and generate embeddings")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.revision or len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision.lower()):
        parser.error("--revision must be a 40-character SFR model commit SHA")
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("batch-size/max-length must be positive")
    if args.device == "cpu" and args.dtype == "float16":
        parser.error("Use float32 on CPU")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.cohort:
        cohort = json.loads(Path(args.cohort).read_text())
        dataset, labels = cohort["metadata"]["dataset"], cohort["labels"]
    elif args.dataset:
        dataset, labels = args.dataset, paper_labels(args.dataset)
    else:
        parser.error("Supply --dataset or --cohort")
    texts = list(dict.fromkeys(text for pair in prompt_texts(dataset, labels) for text in pair))
    # Imports occur only when the user explicitly runs this preparation script.
    import torch
    from transformers import AutoModel, AutoTokenizer
    kwargs = {"revision": args.revision, "local_files_only": args.local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kwargs)
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                     attn_implementation="eager", **kwargs).to(args.device).eval()
    chunks = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        formatted = [f"Instruct: {args.instruction}\nQuery: {text}" for text in batch] if args.format == "instruction" else batch
        tokens = tokenizer(formatted, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(args.device)
        with torch.no_grad():
            hidden = model(**tokens, use_cache=False).last_hidden_state
            indices = tokens["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(len(batch), device=hidden.device), indices]
            chunks.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy())
        print(f"Prompts {min(start + len(batch), len(texts))}/{len(texts)}", flush=True)
    embeddings = np.concatenate(chunks)
    validate_matrix(embeddings, 4096, "SFR prompts")
    metadata = {"model": "Salesforce/SFR-Embedding-Mistral", "model_path": args.model,
        "revision": args.revision, "pooling": "last-token-l2", "format": args.format,
        "instruction": args.instruction if args.format == "instruction" else None,
        "max_length": args.max_length, "dtype": args.dtype, "texts_sha256": stable_hash(texts),
        "recipe_note": "raw matches released get_embed.py SFR branch; paper describes instructions; verify original bank provenance"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, texts=np.array(texts), embeddings=embeddings, metadata=np.array(json.dumps(metadata)))


if __name__ == "__main__":
    main()
