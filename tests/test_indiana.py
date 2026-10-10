import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('indiana', Path(__file__).parents[1] / 'scripts/test_indiana.py')
indiana = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indiana)


def fixture_dataset(root):
    pd.DataFrame([
        {'uid': str(i), 'Problems': 'cardiomegaly;pleural effusion' if i == 1 else 'normal'}
        for i in range(1, 6)
    ]).to_csv(root / 'indiana_reports.csv', index=False)
    views = []
    for i in range(1, 6):
        for view in ('Frontal', 'Lateral'):
            name = f'{i}_{view}.png'
            (root / name).touch()
            views.append({'uid': str(i), 'filename': name, 'projection': view})
    pd.DataFrame(views).to_csv(root / 'indiana_projections.csv', index=False)


def test_subset_is_reproducible_and_study_level(tmp_path):
    fixture_dataset(tmp_path)
    first = indiana.select_subset(tmp_path, 3, 42, 'frontal')
    second = indiana.select_subset(tmp_path, 3, 42, 'frontal')
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == first.uid.nunique() == 3
    assert set(first.projection) == {'Frontal'}
    all_rows = indiana.select_subset(tmp_path, 100, 42, 'all')
    assert len(all_rows) == 5
    positive = all_rows[all_rows.uid == '1'].iloc[0]
    assert positive['label_cardiomegaly'] == positive['label_pleural effusion'] == 1
    assert positive['label_pneumonia'] == 0


def test_missing_image_fails_before_model_loading(tmp_path):
    fixture_dataset(tmp_path)
    (tmp_path / '1_Frontal.png').unlink()
    with pytest.raises(FileNotFoundError):
        indiana.select_subset(tmp_path, 100, 42, 'frontal')
