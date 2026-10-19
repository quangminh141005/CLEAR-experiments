"""PadChest supports native Labels lists and subsets of a full annotation CSV."""
import pandas as pd

from .common import ImageIndex, Record, canonical, find_annotation, finish, tag_list, tag_targets


def load(root, labels=None, **options):
    path = find_annotation(root, "*labels*.csv*", options.get("annotations"))
    frame = pd.read_csv(path, dtype=str)
    required = {"ImageID", "PatientID", options.get("label_column") or "Labels"}
    if not required.issubset(frame.columns):
        raise ValueError(f"PadChest requires columns {sorted(required)}; found {frame.columns.tolist()}")
    field = options.get("label_column") or "Labels"
    aliases = options.get("aliases")
    parsed = [tag_list(x) for x in frame[field]]
    labels = labels or sorted({canonical(t, aliases) for items in parsed if items is not None for t in items})
    index = ImageIndex(root)
    records, missing = [], 0
    for row, items in zip(frame.to_dict("records"), parsed):
        image = index.resolve(row["ImageID"])
        if image is None:
            missing += 1
            continue
        view = str(row.get("Projection", ""))
        if options.get("view", "all") == "frontal" and view.upper() not in {"PA", "AP", "AP_HORIZONTAL"}:
            continue
        if pd.isna(row["PatientID"]) or not str(row["PatientID"]).strip():
            raise ValueError(f"Missing PatientID for {row['ImageID']}")
        target = tag_targets(items, labels, unmentioned=options.get("unmentioned_policy", "unknown"),
                            normal=options.get("normal_policy", "unknown"), aliases=aliases)
        records.append(Record(row["ImageID"], row["PatientID"], str(row.get("StudyID", row["ImageID"])),
                              image, target, items, view))
    return finish(records, labels, {"dataset": "padchest", "annotations": str(path),
        "missing_images": missing, "annotation_rows": len(frame), "patient_ids_verified": True,
        "reference_labels": "Native Labels list; unmentioned findings unknown by default",
        "cohort_warning": "Kaggle sample is not the 7,943-image physician-labelled paper benchmark; annotation provenance must be checked"}, options)
