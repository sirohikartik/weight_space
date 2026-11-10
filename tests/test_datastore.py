"""Unit tests for ActivationDatastore."""

import os
import tempfile
import pytest
from weight_space.datastore import ActivationDatastore


def test_datastore_put_and_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_cache.db")
        ds = ActivationDatastore(db_path=db_path)

        pos_coords = {337: 0.1072, 917: 0.1071}
        neg_coords = {1301: -0.1497, 1434: -0.1416}
        ds.put_recipe("test_behavior", layer_idx=14, alpha=20.0, pos_coords=pos_coords, neg_coords=neg_coords)

        recipe = ds.get_recipe("test_behavior")
        assert recipe is not None
        assert recipe["layer_idx"] == 14
        assert pytest.approx(recipe["alpha"], abs=1e-5) == 20.0
        assert recipe["pos_coords"][337] == 0.1072
        assert recipe["neg_coords"][1301] == -0.1497

        tags = ds.list_recipes()
        assert tags == ["test_behavior"]


def test_datastore_missing_recipe():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_cache.db")
        ds = ActivationDatastore(db_path=db_path)
        assert ds.get_recipe("non_existent") is None
