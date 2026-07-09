"""
Tests for the real-DEM patch sampler (``DEMPatchSampler``).

Uses in-memory / ``.npy`` rasters only -- no network and no ``rasterio``, so the
suite runs anywhere. Checks:
* the ``(batch, generator) -> [B, rows, cols]`` drop-in protocol, shape, dtype,
  device, and exact patch extraction;
* downsampling, augmentation, seed reproducibility;
* nodata (NaN) rejection / filling and referencing (demean / positive);
* ``from_files`` loading and integration with the stratified atmosphere.

Run with::

    pytest test_real_dem.py -v
"""
import numpy as np
import pytest
import torch

from torchdeform.simulation import (
    DEMPatchSampler, load_dem_raster, synthetic_dem,
    copernicus_glo30_url, read_geotiff_bytes,
    ObservationGrid, AtmosphereGenerator, UniformPrior,
)
import torchdeform.simulation.real_dem as real_dem
from torchdeform.atmosphere import stratified_aps


DTYPE = torch.float64
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _ramp_raster(h=200, w=240):
    # distinct value per cell so a patch pins down its own (top, left) offset
    return torch.arange(h * w, dtype=torch.float32).reshape(h, w)


# --------------------------------------------------------------------------- #
# Shape / protocol / extraction
# --------------------------------------------------------------------------- #
class TestBasics:
    def test_shape_dtype(self):
        s = DEMPatchSampler(_ramp_raster(), patch_rows=32, patch_cols=48)
        out = s(5, _gen())
        assert out.shape == (5, 32, 48) and out.dtype == DTYPE

    def test_callable_protocol_matches_synthetic_dem(self):
        # both are (batch, generator) -> [B, rows, cols]
        s = DEMPatchSampler(_ramp_raster(), patch_rows=16, patch_cols=16)
        a = s(3, _gen(1))
        b = synthetic_dem(3, 16, 16, generator=_gen(1))
        assert a.shape == b.shape

    def test_exact_extraction(self):
        # with augment off / downsample 1, a patch is a contiguous sub-block
        r = _ramp_raster(120, 150)
        w = r.shape[1]
        s = DEMPatchSampler(r, patch_rows=20, patch_cols=24, augment=False,
                            max_nan_fraction=1.0, dtype=torch.float32)
        patch = s(1, _gen(3))[0]
        top = int(patch[0, 0].item()) // w
        left = int(patch[0, 0].item()) % w
        torch.testing.assert_close(patch, r[top:top + 20, left:left + 24])

    def test_downsample_extent(self):
        r = _ramp_raster(200, 200)
        s = DEMPatchSampler(r, patch_rows=16, patch_cols=16, downsample=3,
                            augment=False, dtype=torch.float32)
        patch = s(1, _gen(4))[0]
        assert patch.shape == (16, 16)
        # neighbouring output columns are 3 source columns apart
        assert torch.allclose(patch[0, 1] - patch[0, 0], torch.tensor(3.0))


# --------------------------------------------------------------------------- #
# Determinism / augmentation
# --------------------------------------------------------------------------- #
class TestDeterminism:
    def test_seed_reproducible(self):
        s = DEMPatchSampler(_ramp_raster(), patch_rows=24, patch_cols=24)
        torch.testing.assert_close(s(4, _gen(7)), s(4, _gen(7)))

    def test_different_seeds_differ(self):
        s = DEMPatchSampler(_ramp_raster(), patch_rows=24, patch_cols=24)
        assert not torch.allclose(s(4, _gen(0)), s(4, _gen(1)))

    def test_augment_changes_output(self):
        r = _ramp_raster()
        plain = DEMPatchSampler(r, patch_rows=24, patch_cols=24, augment=False)
        aug = DEMPatchSampler(r, patch_rows=24, patch_cols=24, augment=True)
        # same seed, different transform -> generally different patches
        assert not torch.allclose(plain(6, _gen(2)), aug(6, _gen(2)))

    def test_multiple_rasters_used(self):
        # two constant tiles with different values -> a batch spans both
        a = torch.full((80, 80), 10.0)
        b = torch.full((80, 80), 20.0)
        s = DEMPatchSampler([a, b], patch_rows=16, patch_cols=16, augment=False)
        means = s(40, _gen(0)).mean(dim=(-2, -1))
        assert {10.0, 20.0} <= set(round(m.item()) for m in means)


# --------------------------------------------------------------------------- #
# Nodata (NaN) handling
# --------------------------------------------------------------------------- #
class TestNodata:
    def test_output_is_finite_after_fill(self):
        r = _ramp_raster().clone()
        r[:, :120] = float("nan")               # left half nodata
        s = DEMPatchSampler(r, patch_rows=24, patch_cols=24, max_nan_fraction=0.2)
        out = s(8, _gen(1))
        assert torch.isfinite(out).all()

    def test_rejection_prefers_valid_region(self):
        r = _ramp_raster(160, 160).clone()
        r[:, 80:] = float("nan")                # right half nodata
        s = DEMPatchSampler(r, patch_rows=16, patch_cols=16, max_nan_fraction=0.0,
                            augment=False, dtype=torch.float32)
        out = s(16, _gen(5))
        assert torch.isfinite(out).all()        # never forced to keep a nodata patch

    def test_fill_value(self):
        r = _ramp_raster().clone()
        r[::10, ::10] = float("nan")            # nodata lattice: every patch hits some
        s = DEMPatchSampler(r, patch_rows=24, patch_cols=24, augment=False,
                            max_nan_fraction=1.0, fill_value=-999.0,
                            dtype=torch.float32)
        out = s(6, _gen(0))
        assert torch.isfinite(out).all()
        assert (out == -999.0).any()            # leftover nodata filled with fill_value


# --------------------------------------------------------------------------- #
# Referencing
# --------------------------------------------------------------------------- #
class TestReferencing:
    def test_demean(self):
        s = DEMPatchSampler(_ramp_raster(), patch_rows=24, patch_cols=24,
                            augment=False, demean=True)
        out = s(4, _gen(1))
        torch.testing.assert_close(out.mean(dim=(-2, -1)),
                                   torch.zeros(4, dtype=DTYPE), atol=1e-6, rtol=0)

    def test_positive(self):
        s = DEMPatchSampler(_ramp_raster(), patch_rows=24, patch_cols=24,
                            augment=False, positive=True, base_elevation=500.0)
        out = s(4, _gen(2))
        torch.testing.assert_close(out.amin(dim=(-2, -1)),
                                   torch.full((4,), 500.0, dtype=DTYPE),
                                   atol=1e-6, rtol=0)


# --------------------------------------------------------------------------- #
# Errors / construction
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            DEMPatchSampler(torch.zeros(10, 10), patch_rows=32, patch_cols=32)

    def test_no_rasters_raises(self):
        with pytest.raises(ValueError):
            DEMPatchSampler([], patch_rows=8, patch_cols=8)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            DEMPatchSampler(torch.zeros(4, 8, 8), patch_rows=4, patch_cols=4)

    def test_from_files_npy(self, tmp_path):
        p = tmp_path / "tile.npy"
        np.save(p, _ramp_raster(128, 128).numpy())
        s = DEMPatchSampler.from_files(p, patch_rows=32, patch_cols=32)
        assert s(2, _gen()).shape == (2, 32, 32)

    def test_from_files_directory(self, tmp_path):
        for i in range(3):
            np.save(tmp_path / f"tile{i}.npy", (_ramp_raster(96, 96) + i).numpy())
        s = DEMPatchSampler.from_files(tmp_path, patch_rows=24, patch_cols=24)
        assert len(s.rasters) == 3 and s(2, _gen()).shape == (2, 24, 24)

    def test_from_files_empty_raises(self, tmp_path):
        with pytest.raises(ValueError):
            DEMPatchSampler.from_files(tmp_path, patch_rows=8, patch_cols=8)

    def test_load_dem_raster_npy(self, tmp_path):
        p = tmp_path / "r.npy"
        np.save(p, _ramp_raster(20, 30).numpy())
        t = load_dem_raster(p)
        assert t.shape == (20, 30) and t.dtype == torch.float32


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
class TestDevice:
    @pytest.mark.skipif("cuda" not in DEVICES, reason="CUDA not available")
    def test_runs_on_cuda(self):
        s = DEMPatchSampler(_ramp_raster(), patch_rows=16, patch_cols=16,
                            device="cuda")
        out = s(3, torch.Generator(device="cuda").manual_seed(0))
        assert out.device.type == "cuda" and torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Integration: drop-in DEM source
# --------------------------------------------------------------------------- #
class TestIntegration:
    def test_feeds_stratified_aps(self):
        s = DEMPatchSampler(_ramp_raster(), patch_rows=32, patch_cols=40,
                            positive=True, base_elevation=800.0)
        dem = s(2, _gen())
        out = stratified_aps(dem, torch.tensor([3e-3, -2e-3]), model="linear")
        assert out.shape == (2, 32, 40) and torch.isfinite(out).all()

    def test_drop_in_for_atmosphere_generator(self):
        grid = ObservationGrid(rows=32, cols=32)
        sampler = DEMPatchSampler(_ramp_raster(120, 120),
                                  patch_rows=grid.rows, patch_cols=grid.cols)
        atmo = AtmosphereGenerator(grid, strat_coeff=UniformPrior(1e-3, 5e-3),
                                   dem=sampler)
        assert atmo.uses_dem
        screen = atmo.generate(4, generator=_gen(0))
        assert screen.shape == (4, 32, 32) and torch.isfinite(screen).all()


# --------------------------------------------------------------------------- #
# Copernicus GLO-30 (in-memory) -- no network: URL logic is pure, and the read
# path is exercised with a locally built GeoTIFF + a monkeypatched fetch.
# --------------------------------------------------------------------------- #
def _geotiff_bytes(arr, nodata=None):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    arr = np.asarray(arr, dtype=np.float32)
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                     count=1, dtype="float32", nodata=nodata,
                     crs="EPSG:4326",
                     transform=from_origin(14.0, 51.0, 1 / 3600, 1 / 3600)) as dst:
            dst.write(arr, 1)
        return mf.read()


class TestCopernicus:
    def test_url_naming(self):
        assert copernicus_glo30_url(50.4, 14.9).endswith(
            "Copernicus_DSM_COG_10_N50_00_E014_00_DEM.tif")
        # southern / western hemisphere, floored
        assert copernicus_glo30_url(-0.3, -78.6).endswith(
            "Copernicus_DSM_COG_10_S01_00_W079_00_DEM.tif")

    def test_read_geotiff_bytes(self):
        pytest.importorskip("rasterio")
        arr = _ramp_raster(120, 90).numpy()
        arr[0, 0] = -9999.0
        data = _geotiff_bytes(arr, nodata=-9999.0)
        t = read_geotiff_bytes(data)
        assert t.shape == (120, 90) and t.dtype == torch.float32
        assert torch.isnan(t[0, 0])                       # nodata -> NaN

    def test_read_geotiff_bytes_downsample(self):
        pytest.importorskip("rasterio")
        data = _geotiff_bytes(_ramp_raster(400, 400).numpy())
        assert read_geotiff_bytes(data, downsample=4).shape == (100, 100)

    def test_from_copernicus_explicit_tiles(self, monkeypatch):
        pytest.importorskip("rasterio")
        data = _geotiff_bytes(_ramp_raster(300, 300).numpy())
        monkeypatch.setattr(real_dem, "_fetch_bytes", lambda url: data)
        s = DEMPatchSampler.from_copernicus(tiles=[(50, 14), (51, 15)],
                                            patch_rows=64, patch_cols=64)
        assert len(s.rasters) == 2 and s(3, _gen()).shape == (3, 64, 64)

    def test_from_copernicus_random_skips_ocean(self, monkeypatch):
        pytest.importorskip("rasterio")
        data = _geotiff_bytes(_ramp_raster(300, 300).numpy())

        def fake_fetch(url):                              # "ocean" in the west
            if "_W" in url:
                raise FileNotFoundError(url)
            return data
        monkeypatch.setattr(real_dem, "_fetch_bytes", fake_fetch)
        s = DEMPatchSampler.from_copernicus(n=2, seed=0, patch_rows=48, patch_cols=48,
                                            bounds=(0.0, 40.0, -40.0, 40.0))
        assert len(s.rasters) == 2 and s(2, _gen()).shape == (2, 48, 48)

    def test_from_copernicus_exhausted_raises(self, monkeypatch):
        pytest.importorskip("rasterio")
        monkeypatch.setattr(real_dem, "_fetch_bytes",
                            lambda url: (_ for _ in ()).throw(FileNotFoundError(url)))
        with pytest.raises(RuntimeError):
            DEMPatchSampler.from_copernicus(n=1, seed=1, patch_rows=32, patch_cols=32,
                                            max_tile_tries=5)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
