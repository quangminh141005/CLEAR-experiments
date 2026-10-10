#!/usr/bin/env python3
"""Study-level Indiana subset zero-shot evaluation; no training or tuning."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import time
import urllib.request
import zipfile

URL = 'https://www.kaggle.com/api/v1/datasets/download/raddar/chest-xrays-indiana-university'
DISEASES = ('cardiomegaly', 'pleural effusion', 'pneumonia', 'atelectasis', 'pneumothorax')


def prepare(root, download):
    if list(root.rglob('indiana_projections.csv')):
        return
    if not download:
        raise FileNotFoundError(f'Extract the Kaggle dataset in {root} or pass --download')
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'indiana.zip'
    if not archive.exists():
        temporary = root / 'indiana.zip.part'
        print('Downloading full Kaggle archive (subset selected after download)', flush=True)
        try:
            urllib.request.urlretrieve(URL, temporary)
        except Exception as exc:
            raise RuntimeError('Kaggle download failed. Download the supplied URL manually '
                               'or use kaggle datasets download, then extract into --data-dir') from exc
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError('Unsafe archive path')
        bundle.extractall(root)


def select_subset(root, count, seed, projection):
    import pandas as pd
    projections = list(root.rglob('indiana_projections.csv'))
    reports = list(root.rglob('indiana_reports.csv'))
    if len(projections) != 1 or len(reports) != 1:
        raise ValueError('Expected exactly one projections CSV and one reports CSV')
    views = pd.read_csv(projections[0], dtype=str).fillna('')
    tags = pd.read_csv(reports[0], dtype=str).fillna('')
    if not {'uid', 'filename', 'projection'} <= set(views.columns):
        raise ValueError('Projections CSV requires uid, filename, projection')
    if not {'uid', 'Problems'} <= set(tags.columns) or tags.uid.duplicated().any():
        raise ValueError('Reports CSV requires unique uid and Problems tags')
    if projection != 'all':
        views = views[views.projection.str.lower() == projection]
    rows = views.merge(tags, on='uid', validate='many_to_one')
    # One deterministic view per study avoids counting duplicate report labels.
    rows = rows.sort_values(['uid', 'filename']).drop_duplicates('uid')
    uids = sorted(rows.uid.tolist())
    random.Random(seed).shuffle(uids)
    rows = rows.set_index('uid').loc[uids[:count]].reset_index()
    if rows.empty:
        raise ValueError('No eligible studies')
    images = {}
    for path in root.rglob('*.png'):
        if path.name in images:
            raise ValueError(f'Ambiguous image filename: {path.name}')
        images[path.name] = path
    paths = []
    for filename in rows.filename:
        if Path(filename).name not in images:
            raise FileNotFoundError(filename)
        paths.append(str(images[Path(filename).name].resolve()))
    rows['image_path'] = paths
    for disease in DISEASES:
        # Tags are study-level weak labels, not negation-blind report text matches.
        rows['label_' + disease] = rows.Problems.map(
            lambda value: int(disease in {tag.strip().lower() for tag in value.split(';')}))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/indiana'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--subset-size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--projection', choices=['frontal', 'lateral', 'all'], default='frontal')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.subset_size <= 0 or args.batch_size <= 0:
        parser.error('subset-size and batch-size must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'subset.csv').exists():
        parser.error('Output already contains a run; choose a new directory')
    prepare(args.data_dir, args.download)
    rows = select_subset(args.data_dir, args.subset_size, args.seed, args.projection)
    rows.to_csv(args.output / 'subset.csv', index=False)
    if args.prepare_only:
        print(f'Saved {len(rows)} studies to {args.output}/subset.csv')
        return
    os.environ.setdefault('XFORMERS_DISABLED', '1')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    from tqdm import tqdm
    from clear import load_pretrained, tokenize
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Use setup_indiana_k80.sh or explicitly --device cpu')
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            architectures = torch.cuda.get_arch_list()
            if capability == (3, 7) and not any(a in architectures for a in ('sm_37', 'sm_35')):
                raise RuntimeError(f'PyTorch build lacks K80 kernels: {architectures}')
            # Exercise a kernel before expensive model downloads.
            torch.ones(1, device=f'cuda:{index}').sum().item()
    started = time.time()
    model, preprocess = load_pretrained(args.checkpoint, device=args.device)
    model.float()
    prompts = [text for disease in DISEASES for text in
               (f'a chest x-ray showing {disease}', f'a chest x-ray without {disease}')]
    with torch.inference_mode():
        text = F.normalize(model.encode_text(tokenize(prompts).to(args.device)), dim=-1)
        scale = model.logit_scale.exp()
        encoder = model.visual
        if args.device == 'cuda' and torch.cuda.device_count() > 1:
            encoder = torch.nn.DataParallel(encoder)
        results = []
        for start in tqdm(range(0, len(rows), args.batch_size), desc='Indiana inference'):
            batch = []
            for path in rows.image_path.iloc[start:start + args.batch_size]:
                with Image.open(path) as image:
                    batch.append(preprocess(image))
            features = F.normalize(encoder(torch.stack(batch).to(args.device)), dim=-1)
            scores = ((features @ text.T) * scale).reshape(-1, len(DISEASES), 2)
            results.append(scores.softmax(dim=-1)[:, :, 0].cpu().numpy())
    scores = np.concatenate(results)
    metrics = {}
    for index, disease in enumerate(DISEASES):
        labels = rows['label_' + disease].to_numpy()
        rows['score_' + disease] = scores[:, index]
        both = len(np.unique(labels)) == 2
        metrics[disease] = {
            'positive_studies': int(labels.sum()), 'negative_studies': int((1-labels).sum()),
            'auroc': float(roc_auc_score(labels, scores[:, index])) if both else None,
            'average_precision': float(average_precision_score(labels, scores[:, index])) if both else None,
            'f1_at_0.5': float(f1_score(labels, scores[:, index] >= .5, zero_division=0)),
        }
    rows.to_csv(args.output / 'predictions.csv', index=False)
    summary = {
        'studies': len(rows), 'requested_studies': args.subset_size, 'seed': args.seed,
        'projection': args.projection, 'batch_size': args.batch_size,
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                if args.device == 'cuda' else [],
        'checkpoint': str(args.checkpoint) if args.checkpoint else 'peterhan91/CLEAR/best_model.pt',
        'seconds': time.time() - started, 'metrics': metrics,
        'label_source': 'Exact semicolon-separated Problems tags; absent tag treated as negative',
        'score_type': 'Paired-prompt softmax; not calibrated clinical probabilities',
        'limitations': 'Exploratory random subset, study-level weak tags, no training. '
                       'Null ranking metrics indicate a single-class subset.',
    }
    (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))
    print(f'Results saved in {args.output}')


if __name__ == '__main__':
    main()
