"""CheXpert numeric targets preserve missingness and explicit uncertainty."""
import re

import pandas as pd

from .common import ImageIndex, Record, canonical, find_annotation, finish, numeric_target


FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
            "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pleural Other", "Pneumonia",
            "Pneumothorax", "Support Devices", "No Finding"]


def load(root, labels=None, **options):
    # Never concatenate train/valid: users explicitly select another annotation file.
    path = find_annotation(root, "valid.csv", options.get("annotations"))
    frame = pd.read_csv(path, dtype={"Path": str})
    if "Path" not in frame:
        raise ValueError("CheXpert annotation requires Path column")
    aliases = options.get("aliases")
    columns = {canonical(c, aliases): c for c in FINDINGS if c in frame}
    labels = labels or sorted(columns)
    index = ImageIndex(root)
    records, missing = [], 0
    for row in frame.to_dict("records"):
        image = index.resolve(row["Path"])
        if image is None:
            missing += 1
            continue
        view = str(row.get("Frontal/Lateral", ""))
        if options.get("view", "all") == "frontal" and view.lower() != "frontal":
            continue
        match = re.search(r"(?:^|/)(patient[^/]+)/(study[^/]+)", row["Path"].replace("\\", "/"))
        if not match:
            raise ValueError(f"Cannot extract patient/study identity from {row['Path']}")
        target = {label: numeric_target(row.get(columns.get(label)),
                  options.get("uncertain_policy", "ignore"), options.get("missing_policy", "ignore"))
                  for label in labels}
        records.append(Record(row["Path"], match[1], f"{match[1]}/{match[2]}", image, target,
                              {k: None if pd.isna(row.get(v)) else row.get(v) for k, v in columns.items()}, view))
    return finish(records, labels, {"dataset": "chexpert", "annotations": str(path),
        "missing_images": missing, "patient_ids_verified": True,
        "reference_labels": "Numeric 0/1; blank and -1 masked by default",
        "cohort_warning": "Kaggle train/valid is not the 500-patient expert test set. CLEAR pretraining included CheXpert-Plus; train-cohort overlap is possible."}, options)
