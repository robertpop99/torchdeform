"""
Thin ``torch.utils.data.Dataset`` wrappers over the generators.

These map a sample index to a reproducible, **physical** sample (unwrapped
fields + labels) by seeding a per-index RNG (``base_seed + index``), so items are
deterministic and DataLoader-worker-safe. They deliberately do *not* encode ML
targets (normalisation, sin/cos angle encodings, network head layout, ...) --
that is the application's job: pass a ``transform`` callable that turns a sample
into whatever tensors your training loop needs.

For DataLoader training, supply a ``transform`` that returns plain tensors (or a
dict of tensors) so the default collate can batch them; the raw sample objects
(:class:`~torchdeform.simulation.generators.DeformationSample` /
:class:`~torchdeform.simulation.generators.InterferogramSample`) carry mixed
types and are meant for inspection or a custom collate.
"""
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from .generators import (
    DeformationGenerator,
    DeformationSample,
    InterferogramGenerator,
    InterferogramSample,
)


class DeformationDataset(Dataset):
    """Reproducible, index-addressable surface-deformation samples.

    Each ``__getitem__(i)`` seeds a generator with ``base_seed + i`` and draws a
    single sample (batch of 1) from the wrapped
    :class:`~torchdeform.simulation.generators.DeformationGenerator`.

    Parameters
    ----------
    generator : DeformationGenerator
        The deformation generator to draw from.
    length : int
        Number of (virtual) samples; sets ``len(dataset)``.
    base_seed : int
        Per-index RNG offset; item ``i`` always uses seed ``base_seed + i``.
    transform : callable, optional
        Maps a :class:`DeformationSample` to the object ``__getitem__`` returns
        (e.g. your training targets). If ``None``, the raw sample is returned.
    """

    def __init__(
        self,
        generator: DeformationGenerator,
        length: int,
        base_seed: int = 0,
        transform: Optional[Callable[[DeformationSample], object]] = None,
    ):
        self.generator = generator
        self.length = length
        self.base_seed = base_seed
        self.transform = transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        g = torch.Generator().manual_seed(self.base_seed + index)
        sample = self.generator.generate(1, generator=g)
        return self.transform(sample) if self.transform is not None else sample


class InsarDataset(Dataset):
    """Reproducible, index-addressable interferogram samples (full pipeline).

    Like :class:`DeformationDataset`, but draws from an
    :class:`~torchdeform.simulation.generators.InterferogramGenerator`, so each
    sample carries the (unwrapped) deformation phase, optional atmosphere, LOS,
    and all labels. Use ``sample.wrapped()`` for the observable interferogram.

    Parameters
    ----------
    generator : InterferogramGenerator
        The interferogram generator to draw from.
    length : int
        Number of (virtual) samples.
    base_seed : int
        Per-index RNG offset.
    transform : callable, optional
        Maps an :class:`InterferogramSample` to the returned object; ``None``
        returns the raw sample.
    """

    def __init__(
        self,
        generator: InterferogramGenerator,
        length: int,
        base_seed: int = 0,
        transform: Optional[Callable[[InterferogramSample], object]] = None,
    ):
        self.generator = generator
        self.length = length
        self.base_seed = base_seed
        self.transform = transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        g = torch.Generator().manual_seed(self.base_seed + index)
        sample = self.generator.generate(1, generator=g)
        return self.transform(sample) if self.transform is not None else sample
