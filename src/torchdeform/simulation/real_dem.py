"""
Real digital elevation models (DEMs) as a drop-in topography source.

``synthetic_dem`` fabricates terrain procedurally; sometimes you instead want
*real* topography -- for maximally realistic scenes, or to match a specific
region. :class:`DEMPatchSampler` samples random ``[B, rows, cols]`` patches from
one or more real elevation rasters and is **callable with the same
``(batch, generator) -> [B, rows, cols]`` signature as a ``synthetic_dem``
lambda**, so it drops straight into ``AtmosphereGenerator(dem=...)`` /
``InterferogramGenerator`` with nothing else changed::

    sampler = DEMPatchSampler.from_files("dems/", patch_rows=grid.rows,
                                         patch_cols=grid.cols)
    atmo = AtmosphereGenerator(grid, strat_coeff=prior, dem=sampler)

A finite stack of tiles yields effectively unlimited variety via random tile
choice, patch location, flips and 90-degree rotations, and it stays
seed-reproducible (all randomness flows through the passed ``torch.Generator``,
exactly like ``synthetic_dem``). The DEM is input *data*, not a differentiable
model parameter, so patch extraction is (correctly) non-differentiable.

Getting DEMs
------------
Any elevation raster works. Convenient free sources of global ~30 m tiles:

* **Copernicus GLO-30** -- public, no auth. :func:`download_copernicus_glo30`
  fetches one 1x1-degree tile by lat/lon from the open AWS bucket;
  :func:`download_copernicus_glo30_tiles` grabs many at once (an explicit
  ``[(lat, lon), ...]`` list, or ``n`` random land tiles).
* **SRTM / NASADEM / ASTER GDMEM** -- via OpenTopography or NASA Earthdata.

Reading GeoTIFFs needs the optional ``rasterio`` dependency; ``.npy`` / ``.npz``
rasters load with numpy alone.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Union

import torch

from ..core import DeviceLikeType

if TYPE_CHECKING:                                            # annotations only
    import numpy as np

Tensor = torch.Tensor
PathLike = Union[str, os.PathLike]

_RASTER_SUFFIXES = (".tif", ".tiff", ".npy", ".npz")


def _import_rasterio():
    try:
        import rasterio  # type: ignore
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError(
            "Reading GeoTIFF DEMs requires the optional 'rasterio' package "
            "(`pip install 'torchdeform[dem]'` or `pip install rasterio`). For a "
            "dependency-free path, convert the raster to .npy/.npz and load that."
        ) from exc
    return rasterio


def _raster_to_tensor(arr, nodata=None) -> Tensor:
    """Array-like -> contiguous 2-D ``float32`` CPU tensor, ``nodata`` -> ``NaN``.

    Accepts anything ``torch.as_tensor`` understands (incl. numpy arrays returned
    by ``rasterio``/``np.load``), so this module needs no top-level numpy import.
    """
    t = torch.as_tensor(arr).to(torch.float32).cpu().contiguous()
    if nodata is not None:
        t = torch.where(t == float(nodata), t.new_full((), float("nan")), t)
    return t


def load_dem_raster(path: PathLike) -> Tensor:
    """Load one elevation raster to a 2-D ``float32`` tensor, nodata -> ``NaN``.

    Supports GeoTIFF (``.tif``/``.tiff``, via optional ``rasterio``) and numpy
    ``.npy``/``.npz`` (first array, or the ``"dem"`` key if present; needs numpy,
    which those formats imply anyway).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".tif", ".tiff"):
        rasterio = _import_rasterio()
        with rasterio.open(p) as src:
            t = _raster_to_tensor(src.read(1), src.nodata)
    elif suffix in (".npy", ".npz"):
        import numpy as np                                   # numpy's own formats
        if suffix == ".npy":
            arr = np.load(p)
        else:
            z = np.load(p)
            arr = z["dem" if "dem" in z.files else z.files[0]]
        t = _raster_to_tensor(arr)
    else:
        raise ValueError(f"unsupported DEM raster {p.name!r}; expected one of "
                         f"{_RASTER_SUFFIXES}")
    if t.ndim != 2:
        raise ValueError(f"DEM raster {p.name!r} is {t.ndim}-D; expected a 2-D grid")
    return t


def _as_raster_tensor(r) -> Tensor:
    t = _raster_to_tensor(r)
    if t.ndim != 2:
        raise ValueError(f"each raster must be 2-D [rows, cols]; got shape {tuple(t.shape)}")
    return t


class DEMPatchSampler:
    """Sample random ``[B, rows, cols]`` elevation patches from real DEM raster(s).

    Instances are callable as ``sampler(batch, generator) -> [B, rows, cols]``,
    matching the ``synthetic_dem`` lambda protocol expected by
    ``AtmosphereGenerator(dem=...)``.

    Parameters
    ----------
    rasters : tensor | ndarray | sequence of them
        One or more 2-D elevation grids (metres). ``NaN`` marks nodata (e.g. sea
        or voids); such pixels are avoided/filled (see ``max_nan_fraction``).
    patch_rows, patch_cols : int
        Output patch size.
    downsample : int
        Take every ``downsample``-th pixel, so a patch spans
        ``patch * downsample`` source pixels (coarser resolution, wider extent).
    augment : bool
        Random horizontal/vertical flips (and 90-degree rotations when the patch
        is square) to multiply the variety of a small tile stack.
    max_nan_fraction : float
        Reject-and-resample patches whose nodata fraction exceeds this (up to
        ``max_tries``); remaining ``NaN`` are then filled (``fill_value`` or the
        patch mean). Set to ``1.0`` to disable rejection.
    fill_value : float, optional
        Value for leftover nodata; ``None`` (default) uses the per-patch mean.
    demean : bool
        Subtract each patch's mean (relief about zero).
    positive : bool
        Shift each patch so its minimum equals ``base_elevation``. Mutually
        exclusive with ``demean`` (``demean`` wins if both are set).
    base_elevation : float
        Reference for ``positive``.
    device, dtype :
        Device and dtype of the returned patches (rasters are held on CPU as
        ``float32`` regardless, to keep memory modest).
    """

    def __init__(
        self,
        rasters: Union[Tensor, np.ndarray, Sequence[Union[Tensor, np.ndarray]]],
        *,
        patch_rows: int,
        patch_cols: int,
        downsample: int = 1,
        augment: bool = True,
        max_nan_fraction: float = 0.2,
        fill_value: Optional[float] = None,
        demean: bool = False,
        positive: bool = False,
        base_elevation: float = 0.0,
        max_tries: int = 20,
        device: Optional[DeviceLikeType] = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        if torch.is_tensor(rasters) or hasattr(rasters, "__array__"):
            rasters = [rasters]                              # a single raster, not a stack
        self.rasters: List[Tensor] = [_as_raster_tensor(r) for r in rasters]
        if not self.rasters:
            raise ValueError("no rasters supplied")

        self.patch_rows = int(patch_rows)
        self.patch_cols = int(patch_cols)
        self.downsample = int(downsample)
        self.augment = bool(augment)
        self.max_nan_fraction = float(max_nan_fraction)
        self.fill_value = fill_value
        self.demean = bool(demean)
        self.positive = bool(positive)
        self.base_elevation = float(base_elevation)
        self.max_tries = int(max_tries)
        self.device = device
        self.dtype = dtype

        self._extent_r = self.patch_rows * self.downsample
        self._extent_c = self.patch_cols * self.downsample
        self._eligible = [i for i, r in enumerate(self.rasters)
                          if r.shape[0] >= self._extent_r and r.shape[1] >= self._extent_c]
        if not self._eligible:
            raise ValueError(
                f"no raster is large enough for a {self.patch_rows}x{self.patch_cols} "
                f"patch at downsample={self.downsample} "
                f"(needs >= {self._extent_r}x{self._extent_c} pixels)")

    # -- internals ---------------------------------------------------------- #
    def _randint(self, high: int, generator, gdev) -> int:
        if high <= 1:
            return 0
        return int(torch.randint(high, (1,), generator=generator, device=gdev).item())

    def _extract_one(self, generator, gdev) -> Tensor:
        best = None
        for _ in range(max(1, self.max_tries)):
            ridx = self._eligible[self._randint(len(self._eligible), generator, gdev)]
            r = self.rasters[ridx]
            top = self._randint(r.shape[0] - self._extent_r + 1, generator, gdev)
            left = self._randint(r.shape[1] - self._extent_c + 1, generator, gdev)
            patch = r[top:top + self._extent_r:self.downsample,
                      left:left + self._extent_c:self.downsample]
            nan_frac = torch.isnan(patch).float().mean().item()
            if best is None or nan_frac < best[1]:
                best = (patch, nan_frac)
            if nan_frac <= self.max_nan_fraction:
                break
        return best[0].clone()

    def _augment(self, patch: Tensor, generator, gdev) -> Tensor:
        if self._randint(2, generator, gdev):
            patch = patch.flip(-1)
        if self._randint(2, generator, gdev):
            patch = patch.flip(-2)
        if self.patch_rows == self.patch_cols:
            k = self._randint(4, generator, gdev)
            if k:
                patch = torch.rot90(patch, k, dims=(-2, -1))
        return patch

    def _finalize(self, patch: Tensor) -> Tensor:
        nan = torch.isnan(patch)
        if nan.any():
            if self.fill_value is not None:
                fill = self.fill_value
            else:
                valid = patch[~nan]
                fill = valid.mean() if valid.numel() else patch.new_zeros(())
            patch = torch.where(nan, torch.as_tensor(fill, dtype=patch.dtype), patch)
        if self.demean:
            patch = patch - patch.mean()
        elif self.positive:
            patch = patch - patch.min() + self.base_elevation
        return patch

    # -- public ------------------------------------------------------------- #
    def __call__(self, batch: int,
                 generator: Optional[torch.Generator] = None) -> Tensor:
        """Return ``[batch, patch_rows, patch_cols]`` real-terrain patches (metres)."""
        gdev = generator.device if generator is not None else torch.device("cpu")
        patches = []
        for _ in range(int(batch)):
            patch = self._extract_one(generator, gdev)
            if self.augment:
                patch = self._augment(patch, generator, gdev)
            patches.append(self._finalize(patch))
        out = torch.stack(patches, dim=0)
        return out.to(device=self.device, dtype=self.dtype)

    # -- construction helpers ---------------------------------------------- #
    @classmethod
    def from_files(cls, paths: Union[PathLike, Iterable[PathLike]], **kwargs
                   ) -> "DEMPatchSampler":
        """Build a sampler from raster files or a directory of them.

        ``paths`` is a directory (all ``.tif/.tiff/.npy/.npz`` inside are loaded),
        a single file, or an iterable of files. Remaining keyword arguments are
        forwarded to :class:`DEMPatchSampler`.
        """
        files: List[Path] = []
        if isinstance(paths, (str, os.PathLike)):
            p = Path(paths)
            if p.is_dir():
                files = sorted(f for f in p.iterdir()
                               if f.suffix.lower() in _RASTER_SUFFIXES)
            else:
                files = [p]
        else:
            files = [Path(f) for f in paths]
        if not files:
            raise ValueError(f"no DEM rasters found at {paths!r}")
        rasters = [load_dem_raster(f) for f in files]
        return cls(rasters, **kwargs)

    @classmethod
    def from_copernicus(
        cls,
        n: int = 1,
        *,
        patch_rows: int,
        patch_cols: int,
        tiles: Optional[Sequence[tuple]] = None,
        seed: Optional[int] = None,
        bounds: tuple = (-56.0, 60.0, -180.0, 180.0),
        read_downsample: int = 1,
        max_tile_tries: int = 200,
        url_for=None,
        **kwargs,
    ) -> "DEMPatchSampler":
        """Build a sampler from Copernicus GLO-30 tiles fetched **into memory** (no disk).

        Either pass explicit ``tiles`` as ``[(lat, lon), ...]``, or leave it
        ``None`` to draw ``n`` random *land* tiles: random ``(lat, lon)`` within
        ``bounds = (lat_min, lat_max, lon_min, lon_max)`` are tried, skipping the
        oceans (404s), until ``n`` tiles are collected or ``max_tile_tries`` is
        exhausted. Explicit ocean tiles are likewise skipped (with a
        :mod:`warnings` message, since each was named on purpose). Tile choice is
        reproducible via ``seed`` (the per-patch sampling still uses the
        ``torch.Generator`` passed at call time).

        ``url_for(lat, lon) -> url`` (default :func:`copernicus_glo30_url`) builds
        the tile URL; override it to fetch from a differently-named source.

        Each tile is ~3600x3600 px (~52 MB float32 near the equator);
        ``read_downsample`` reads a decimated overview to cut that ~k^2. Requires
        network access and the optional ``rasterio`` dependency. Untested in CI.
        """
        url_for = url_for or copernicus_glo30_url

        def _fetch(la, lo):
            return read_geotiff_bytes(_fetch_bytes(url_for(la, lo)), read_downsample)

        if tiles is not None:
            rasters = _fetch_tiles(tiles, _fetch)
        else:
            rasters = _collect_land_tiles(n, bounds, seed, max_tile_tries, _fetch)
        return cls(rasters, patch_rows=patch_rows, patch_cols=patch_cols, **kwargs)


def copernicus_glo30_url(lat: float, lon: float) -> str:
    """Public AWS URL of the Copernicus GLO-30 tile covering ``floor(lat), floor(lon)``."""
    la, lo = int(math.floor(lat)), int(math.floor(lon))
    ns, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
    name = f"Copernicus_DSM_COG_10_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def _fetch_tiles(tiles, fetch):
    """``[fetch(lat, lon) for (lat, lon) in tiles]``, skipping ocean tiles (404s).

    Unlike the random-draw path, an explicit list implies the caller wants every
    tile, so each skipped one is announced via :mod:`warnings`.
    """
    import warnings

    out = []
    for la, lo in tiles:
        try:
            out.append(fetch(la, lo))
        except FileNotFoundError:
            warnings.warn(f"skipping tile lat={la}, lon={lo}: no Copernicus GLO-30 "
                          f"tile there (ocean or out of coverage)", stacklevel=2)
    return out


def _collect_land_tiles(n, bounds, seed, max_tile_tries, fetch):
    """Draw up to ``n`` random *land* tiles, calling ``fetch(lat, lon)`` per tile.

    Random integer ``(lat, lon)`` keys are drawn (reproducibly via ``seed``)
    within ``bounds = (lat_min, lat_max, lon_min, lon_max)``, deduplicated, and
    passed to ``fetch``; keys where ``fetch`` raises :class:`FileNotFoundError`
    are treated as ocean and skipped. Raises :class:`RuntimeError` if fewer than
    ``n`` land tiles are found within ``max_tile_tries`` draws.
    """
    import random

    rng = random.Random(seed)
    lat_min, lat_max, lon_min, lon_max = bounds
    results, seen, tries = [], set(), 0
    while len(results) < n and tries < max_tile_tries:
        tries += 1
        key = (int(math.floor(rng.uniform(lat_min, lat_max))),
               int(math.floor(rng.uniform(lon_min, lon_max))))
        if key in seen:
            continue
        seen.add(key)
        try:
            results.append(fetch(*key))
        except FileNotFoundError:
            continue                                             # ocean / no tile
    if len(results) < n:
        raise RuntimeError(
            f"found only {len(results)}/{n} land tiles in {tries} tries; "
            f"widen `bounds`, raise `max_tile_tries`, or pass explicit `tiles`")
    return results


def _fetch_bytes(url: str) -> bytes:
    """GET ``url`` into memory; raise :class:`FileNotFoundError` on 404 (no tile)."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url) as resp:                # noqa: S310 (fixed host)
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(f"no tile at {url} (ocean or out of coverage)") from exc
        raise


def read_geotiff_bytes(data: bytes, downsample: int = 1) -> Tensor:
    """Decode an in-memory GeoTIFF (COG) to a 2-D ``float32`` tensor, nodata -> ``NaN``.

    ``downsample`` reads a decimated overview (``out_shape``), cutting the read
    cost and memory footprint ~``downsample**2`` -- handy for holding several
    tiles in RAM. Needs the optional ``rasterio`` dependency.
    """
    rasterio = _import_rasterio()
    with rasterio.io.MemoryFile(data) as mf, mf.open() as src:
        if downsample > 1:
            out = (max(1, src.height // downsample), max(1, src.width // downsample))
            arr = src.read(1, out_shape=out)
        else:
            arr = src.read(1)
        nodata = src.nodata
    return _raster_to_tensor(arr, nodata)


def download_copernicus_glo30(lat: float, lon: float, *,
                              cache_dir: PathLike = "~/.cache/torchdeform/dem",
                              overwrite: bool = False,
                              url_for=None) -> Path:
    """Download one Copernicus GLO-30 1x1-degree DEM tile to disk and return its path.

    Fetches from the public (no-auth) AWS bucket ``copernicus-dem-30m`` the tile
    covering ``floor(lat), floor(lon)``. Ocean-only tiles do not exist and raise
    :class:`FileNotFoundError`. ``url_for(lat, lon) -> url`` (default
    :func:`copernicus_glo30_url`) builds the URL; override it to fetch from a
    differently-named source. Requires network access; read the result with
    :meth:`DEMPatchSampler.from_files`. For an in-memory alternative (no disk),
    see :meth:`DEMPatchSampler.from_copernicus`; for many tiles at once, see
    :func:`download_copernicus_glo30_tiles`. Untested in CI.
    """
    import urllib.error
    import urllib.request

    url = (url_for or copernicus_glo30_url)(lat, lon)
    cache = Path(os.path.expanduser(str(cache_dir)))
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / url.rsplit("/", 1)[-1]
    if dest.exists() and not overwrite:
        return dest
    try:
        urllib.request.urlretrieve(url, dest)                    # noqa: S310 (fixed host)
    except urllib.error.HTTPError as exc:
        if dest.exists():
            dest.unlink()
        if exc.code == 404:
            raise FileNotFoundError(
                f"no Copernicus GLO-30 tile at lat={int(math.floor(lat))}, "
                f"lon={int(math.floor(lon))} (ocean or out of coverage): {url}") from exc
        raise
    return dest


def download_copernicus_glo30_tiles(
    tiles: Optional[Sequence[tuple]] = None,
    *,
    n: int = 1,
    bounds: tuple = (-56.0, 60.0, -180.0, 180.0),
    seed: Optional[int] = None,
    max_tile_tries: int = 200,
    cache_dir: PathLike = "~/.cache/torchdeform/dem",
    overwrite: bool = False,
    url_for=None,
) -> List[Path]:
    """Download several Copernicus GLO-30 tiles to disk; return their paths.

    The bulk / to-disk counterpart of :func:`download_copernicus_glo30` (which it
    calls per tile). Either pass explicit ``tiles`` as ``[(lat, lon), ...]``, or
    leave it ``None`` to grab ``n`` random *land* tiles within
    ``bounds = (lat_min, lat_max, lon_min, lon_max)`` (reproducible via ``seed``,
    up to ``max_tile_tries`` draws). Ocean tiles (404s) are skipped in **both**
    modes, so an explicit list returns only the paths that existed -- each skipped
    explicit tile is announced via :mod:`warnings`. ``cache_dir``, ``overwrite``
    and ``url_for`` are passed through per tile.

    For an in-memory alternative that skips disk entirely, see
    :meth:`DEMPatchSampler.from_copernicus`. Requires network access. Untested
    in CI.
    """
    def _fetch(la, lo):
        return download_copernicus_glo30(la, lo, cache_dir=cache_dir,
                                         overwrite=overwrite, url_for=url_for)

    if tiles is not None:
        return _fetch_tiles(tiles, _fetch)
    return _collect_land_tiles(n, bounds, seed, max_tile_tries, _fetch)
