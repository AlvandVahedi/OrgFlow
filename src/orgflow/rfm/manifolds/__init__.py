"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from orgflow.rfm.manifolds.analog_bits import MultiAtomAnalogBits
from orgflow.rfm.manifolds.euclidean import EuclideanWithLogProb
from orgflow.rfm.manifolds.flat_torus import (
    FlatTorus01FixFirstAtomToOrigin,
    FlatTorus01FixFirstAtomToOriginWrappedNormal,
)
from orgflow.rfm.manifolds.null import NullManifoldWithDeltaRandom
from orgflow.rfm.manifolds.product import ProductManifoldWithLogProb
from orgflow.rfm.manifolds.simplex import (
    FlatDirichletSimplex,
    MultiAtomFlatDirichletSimplex,
)
