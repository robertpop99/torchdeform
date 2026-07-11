"""
Live Copernicus GLO-30 download tests (manual-only).

These actually hit the public AWS bucket ``copernicus-dem-30m`` and decode a real
GeoTIFF, so they need **network access** and the optional ``rasterio`` dependency.
They are the one check the CI mocks (``tests/simulation/test_real_dem.py``, which
monkeypatch the fetch) cannot do: that the real tile-URL scheme, the AWS bucket,
and rasterio's COG decoding still line up. If Copernicus renames its tiles or
moves the bucket, only a live download notices.

The suite is skipped unless ``RUN_COPERNICUS_TESTS=1`` is set, so a plain
``pytest`` / CI run never touches it::

    RUN_COPERNICUS_TESTS=1 pytest tests/external/test_copernicus_download.py -v -s

Each GLO-30 tile is ~1x1 degree (~3600x3600 px, tens of MB), so the suite keeps
the tile count small and downloads into a throwaway ``tmp_path`` cache.
"""
from __future__ import annotations

import os

import pytest
import torch

# --------------------------------------------------------------------------- #
# Gating: manual-only + rasterio availability
# --------------------------------------------------------------------------- #
if os.environ.get("RUN_COPERNICUS_TESTS") != "1":
    pytest.skip(
        "Copernicus download is a manual suite; set RUN_COPERNICUS_TESTS=1 to run it.",
        allow_module_level=True,
    )

pytest.importorskip("rasterio")

from torchdeform.simulation import (               # noqa: E402  (after the gate)
    DEMPatchSampler,
    download_copernicus_glo30,
    download_copernicus_glo30_tiles,
    read_geotiff_bytes,
)


# A known-good land tile (Czech/Saxony border, floors to N50 E14) and an
# open-ocean point (mid-Pacific, floors to N00 W140 -- no tile exists there).
LAND = (50.4, 14.9)
OCEAN = (0.0, -140.0)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


class TestLiveDownload:
    def test_land_tile_downloads_to_disk(self, tmp_path):
        p = download_copernicus_glo30(*LAND, cache_dir=tmp_path)
        assert p.exists() and p.stat().st_size > 0
        t = read_geotiff_bytes(p.read_bytes())          # decodes as a real raster
        assert t.ndim == 2 and torch.isfinite(t).any()

    def test_ocean_tile_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            download_copernicus_glo30(*OCEAN, cache_dir=tmp_path)

    def test_download_tiles_explicit_skips_ocean(self, tmp_path):
        # land + ocean -> ocean 404 is skipped (with a warning), land kept
        with pytest.warns(UserWarning, match="skipping tile"):
            paths = download_copernicus_glo30_tiles([LAND, OCEAN], cache_dir=tmp_path)
        assert len(paths) == 1 and paths[0].exists()

    def test_download_tiles_random_land(self, tmp_path):
        # bounds entirely over central Europe -> a random draw lands on real tiles fast
        paths = download_copernicus_glo30_tiles(
            n=1, seed=0, bounds=(45.0, 52.0, 6.0, 16.0), cache_dir=tmp_path)
        assert len(paths) == 1 and paths[0].exists()
        s = DEMPatchSampler.from_files(paths[0], patch_rows=64, patch_cols=64)
        assert s(2, _gen()).shape == (2, 64, 64)

    def test_from_copernicus_in_memory(self):
        # downsample keeps the decoded raster small; the fetch is still live
        s = DEMPatchSampler.from_copernicus(
            tiles=[LAND], patch_rows=64, patch_cols=64, read_downsample=8)
        out = s(2, _gen())
        assert out.shape == (2, 64, 64) and torch.isfinite(out).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
