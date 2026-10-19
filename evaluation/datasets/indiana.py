"""Indiana Problems tags are weak study-level labels, copied to each image."""
import pandas as pd

from .common import (Cohort, ImageIndex, Record, canonical, find_annotation, finish,
                     patient_mapping, tag_list, tag_targets)


def load(root, labels=None, **options):
    reports_path = find_annotation(root, "indiana_reports.csv", options.get("annotations"))
    views_path = find_annotation(root, "indiana_projections.csv", options.get("projections"))
    reports = pd.read_csv(reports_path, dtype={"uid": str})
    views = pd.read_csv(views_path, dtype={"uid": str})
    if reports.uid.duplicated().any():
        raise ValueError("Duplicate Indiana report uid")
    field = options.get("label_column") or "Problems"
    aliases = options.get("aliases")
    if field not in reports:
        raise ValueError(f"Missing Indiana label column {field}")
    tags = {uid: tag_list(value) for uid, value in zip(reports.uid, reports[field])}
    labels = labels or sorted({canonical(t, aliases) for items in tags.values() if items is not None for t in items})
    index = ImageIndex(root)
    mapping = patient_mapping(options.get("patient_map"))
    records, missing = [], 0
    for row in views.to_dict("records"):
        uid = str(row["uid"])
        view = str(row.get("projection", ""))
        if options.get("view", "all") == "frontal" and view.lower() not in {"frontal", "pa", "ap"}:
            continue
        image = index.resolve(row["filename"])
        if image is None:
            missing += 1
            continue
        if mapping is not None and uid not in mapping:
            raise ValueError(f"Missing patient-map entry for Indiana study {uid}")
        items = tags.get(uid)
        target = tag_targets(items, labels, unmentioned=options.get("unmentioned_policy", "unknown"),
                             normal=options.get("normal_policy", "unknown"), aliases=aliases)
        records.append(Record(f"{uid}/{row['filename']}", mapping[uid] if mapping else uid,
                              uid, image, target, items, view))
    return finish(records, labels, {"dataset": "indiana", "annotations": str(reports_path),
        "projections": str(views_path), "missing_images": missing, "patient_ids_verified": mapping is not None,
        "grouping": "patient-map" if mapping else "uid study proxy; not verified patient identity",
        "reference_labels": "Problems tags; unannotated findings unknown unless explicitly configured"}, options)
