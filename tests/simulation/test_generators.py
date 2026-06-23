"""
Tests for the composition layer (generators) and the Dataset wrappers.

Covers the observation grid, single-type and mixture deformation generation
(grouping, weighting, reproducibility), geometry generation (single prior and
mixture), the atmosphere composition, the full interferogram pipeline (unwrapped
storage + wrapped-on-demand), and the index-addressable datasets.

Run with::

    pytest test_generators.py -v
"""
import math

import pytest
import torch

from torchdeform import (
    MogiSource, PennySource, OkadaSourceSimple, okada_params_from_fault,
)
from torchdeform.simulation import (
    ObservationGrid, centered_location, SourceGenerator, DeformationGenerator,
    DeformationSample, GeometryGenerator, AtmosphereGenerator,
    InterferogramGenerator, InterferogramSample,
    DeformationDataset, InsarDataset,
    GeometryPrior, PriorMixture, UniformPrior, ConstantPrior, LocationPrior,
    DEFAULT_MOGI_PRIOR, DEFAULT_PENNY_PRIOR, DEFAULT_EARTHQUAKE_PRIOR,
    DEFAULT_S1_GEOMETRY_PRIOR, synthetic_dem,
)


DTYPE = torch.float64


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _grid():
    return ObservationGrid(32, 32, psizex=500.0, psizey=500.0, dtype=DTYPE)


def _mogi_gen(grid):
    return DeformationGenerator(grid, {"mogi": SourceGenerator(MogiSource(), DEFAULT_MOGI_PRIOR)})


# --------------------------------------------------------------------------- #
# ObservationGrid
# --------------------------------------------------------------------------- #
def test_grid_shape_and_centering():
    grid = ObservationGrid(4, 6, psizex=2.0, psizey=3.0, dtype=DTYPE)
    assert grid.n == 24
    assert grid.extent == ((6 - 1) * 2.0, (4 - 1) * 3.0)
    x, y = grid.coords(5)
    assert x.shape == (5, 24) and y.shape == (5, 24)
    # centred on (0, 0)
    assert torch.allclose(x.mean(), torch.zeros(1, dtype=DTYPE), atol=1e-9)
    assert torch.allclose(y.mean(), torch.zeros(1, dtype=DTYPE), atol=1e-9)


def test_centered_location_within_extent():
    grid = _grid()
    loc = centered_location(grid, frac=0.5)
    out = loc.sample((1000,), _gen())
    ex, ey = grid.extent
    assert out["source_x"].abs().max() <= frac_bound(ex, 0.5)
    assert out["source_y"].abs().max() <= frac_bound(ey, 0.5)


def frac_bound(extent, frac):
    return frac * extent / 2.0 + 1e-6


# --------------------------------------------------------------------------- #
# SourceGenerator
# --------------------------------------------------------------------------- #
def test_source_generator_mogi():
    grid = _grid()
    x, y = grid.coords(8)
    sg = SourceGenerator(MogiSource(), DEFAULT_MOGI_PRIOR)
    disp, params = sg.generate(x, y, torch.zeros(8, dtype=DTYPE), torch.zeros(8, dtype=DTYPE),
                               generator=_gen())
    assert disp.u.shape == (8, grid.n)
    assert torch.isfinite(disp.u).all()
    assert set(params) == {"depth", "delta_v"}          # location excluded from labels
    assert "source_x" not in params


def test_source_generator_okada():
    grid = _grid()
    x, y = grid.coords(4)
    sg = SourceGenerator(OkadaSourceSimple(training_safe=True), DEFAULT_EARTHQUAKE_PRIOR,
                         to_forward=okada_params_from_fault)
    disp, params = sg.generate(x, y, torch.zeros(4, dtype=DTYPE), torch.zeros(4, dtype=DTYPE),
                               generator=_gen())
    assert disp.u.shape == (4, grid.n)
    assert torch.isfinite(disp.u).all()
    assert "strike" in params and "centroid_depth" not in params   # labels are geophysical


# --------------------------------------------------------------------------- #
# DeformationGenerator
# --------------------------------------------------------------------------- #
def test_deformation_generator_groups_partition_batch():
    grid = _grid()
    gen = DeformationGenerator(grid, {
        "mogi": SourceGenerator(MogiSource(), DEFAULT_MOGI_PRIOR),
        "penny": SourceGenerator(PennySource(), DEFAULT_PENNY_PRIOR),
    })
    out = gen.generate(64, generator=_gen())
    assert isinstance(out, DeformationSample)
    assert out.displacement.u.shape == (64, grid.n)
    assert len(out.source_type) == 64
    all_idx = torch.cat([out.index[t] for t in out.index])
    assert torch.equal(all_idx.sort().values, torch.arange(64))   # exact cover
    for t, idx in out.index.items():
        assert all(out.source_type[i] == t for i in idx.tolist())


def test_deformation_generator_weighting():
    grid = _grid()
    gen = DeformationGenerator(grid, {
        "mogi": SourceGenerator(MogiSource(), DEFAULT_MOGI_PRIOR),
        "penny": SourceGenerator(PennySource(), DEFAULT_PENNY_PRIOR),
    }, weights={"mogi": 9.0, "penny": 1.0})
    out = gen.generate(4000, generator=_gen())
    frac_mogi = sum(t == "mogi" for t in out.source_type) / 4000
    assert 0.85 < frac_mogi < 0.95


def test_deformation_generator_reproducible():
    grid = _grid()
    gen = _mogi_gen(grid)
    a = gen.generate(16, generator=_gen(7))
    b = gen.generate(16, generator=_gen(7))
    assert torch.allclose(a.displacement.u, b.displacement.u)
    assert torch.allclose(a.source_x, b.source_x)


def test_deformation_generator_validation():
    grid = _grid()
    with pytest.raises(ValueError):
        DeformationGenerator(grid, {})
    with pytest.raises(ValueError):
        DeformationGenerator(grid, {"mogi": SourceGenerator(MogiSource(), DEFAULT_MOGI_PRIOR)},
                             weights={"penny": 1.0})


# --------------------------------------------------------------------------- #
# GeometryGenerator
# --------------------------------------------------------------------------- #
def test_geometry_generator_single_prior():
    gg = GeometryGenerator(DEFAULT_S1_GEOMETRY_PRIOR)
    los, g = gg.generate(16, generator=_gen(), dtype=DTYPE)
    assert los.e.shape == (16, 1)
    assert torch.allclose(los.norm, torch.ones_like(los.norm), atol=1e-12)
    assert set(g) >= {"heading_deg", "incidence_deg", "look_side"}


def test_geometry_generator_mixture_flattens_to_batch():
    asc = GeometryPrior(UniformPrior(-15.0, -13.0), UniformPrior(29.0, 46.0))
    desc = GeometryPrior(UniformPrior(193.0, 195.0), UniformPrior(29.0, 46.0))
    gg = GeometryGenerator(PriorMixture({"asc": asc, "desc": desc}))
    los, g = gg.generate(2000, generator=_gen(), dtype=DTYPE)
    assert los.e.shape == (2000, 1)
    assert g["heading_deg"].shape == (2000,)
    h = g["heading_deg"]
    in_asc = (h >= -15) & (h <= -13)
    in_desc = (h >= 193) & (h <= 195)
    assert bool((in_asc | in_desc).all())
    assert in_asc.any() and in_desc.any()


# --------------------------------------------------------------------------- #
# AtmosphereGenerator
# --------------------------------------------------------------------------- #
def test_atmosphere_empty_is_zero():
    grid = _grid()
    out = AtmosphereGenerator(grid).generate(4, generator=_gen())
    assert out.shape == (4, grid.rows, grid.cols)
    assert torch.all(out == 0)


def test_atmosphere_components_finite():
    grid = _grid()
    atm = AtmosphereGenerator(
        grid, orbital_rms=UniformPrior(2.0, 5.0), turbulent_rms=UniformPrior(0.5, 1.5),
        strat_coeff=UniformPrior(-3e-3, 3e-3),
        dem=lambda b, g: synthetic_dem(b, grid.rows, grid.cols, relief=600.0, generator=g),
    )
    out = atm.generate(4, generator=_gen())
    assert out.shape == (4, grid.rows, grid.cols)
    assert torch.isfinite(out).all() and out.abs().sum() > 0


def test_atmosphere_stratified_needs_dem():
    grid = _grid()
    with pytest.raises(ValueError, match="dem"):
        AtmosphereGenerator(grid, strat_coeff=UniformPrior(-3e-3, 3e-3)).generate(2)


# --------------------------------------------------------------------------- #
# InterferogramGenerator
# --------------------------------------------------------------------------- #
def _ifg_gen(grid, with_atm=True):
    defo = _mogi_gen(grid)
    geom = GeometryGenerator(DEFAULT_S1_GEOMETRY_PRIOR)
    atm = AtmosphereGenerator(grid, orbital_rms=UniformPrior(2.0, 5.0)) if with_atm else None
    return InterferogramGenerator(defo, geom, atm)


def test_interferogram_sample_fields_and_wrap():
    grid = _grid()
    s = _ifg_gen(grid).generate(8, generator=_gen())
    assert isinstance(s, InterferogramSample)
    assert s.deformation_phase.shape == (8, grid.rows, grid.cols)
    assert s.atmosphere.shape == (8, grid.rows, grid.cols)
    # phase is the unwrapped sum; wrapped() is the observable in [-pi, pi]
    assert torch.allclose(s.phase, s.deformation_phase + s.atmosphere)
    w = s.wrapped()
    assert w.shape == (8, grid.rows, grid.cols)
    assert bool((w.abs() <= math.pi + 1e-6).all())


def test_interferogram_without_atmosphere():
    grid = _grid()
    s = _ifg_gen(grid, with_atm=False).generate(4, generator=_gen())
    assert s.atmosphere is None
    assert torch.allclose(s.phase, s.deformation_phase)


def test_interferogram_reproducible():
    grid = _grid()
    gen = _ifg_gen(grid)
    a = gen.generate(4, generator=_gen(3))
    b = gen.generate(4, generator=_gen(3))
    assert torch.allclose(a.wrapped(), b.wrapped())


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def test_deformation_dataset_len_item_and_reproducible():
    grid = _grid()
    ds = DeformationDataset(_mogi_gen(grid), length=50)
    assert len(ds) == 50
    item = ds[5]
    assert item.displacement.u.shape == (1, grid.n)        # batch dim 1
    # deterministic per index, independent of access order
    assert torch.allclose(ds[5].displacement.u, ds[5].displacement.u)
    assert not torch.allclose(ds[5].displacement.u, ds[6].displacement.u)


def test_insar_dataset_and_transform():
    grid = _grid()
    gen = _ifg_gen(grid)
    ds = InsarDataset(gen, length=20)
    assert ds[3].wrapped().shape == (1, grid.rows, grid.cols)

    # transform hook turns a sample into whatever the training loop needs
    ds_t = InsarDataset(gen, length=20, transform=lambda s: s.wrapped().squeeze(0))
    out = ds_t[3]
    assert out.shape == (grid.rows, grid.cols)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
