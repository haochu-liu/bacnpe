"""
Regression ("golden value") tests.

The expected vectors were produced by the original, pre-package code
(with only the Tajima's D index bug fixed so that it runs) using the same
seeds. They guard against any future change of the simulation output.
NumPy's legacy global random stream (np.random.seed) is frozen across NumPy
versions, so these values are stable; the exponential-fit coefficients
(entries 26-28) come from scipy's optimiser and get a small tolerance.
"""
import numpy as np
import pytest

from bacnpe import ClonalOrigin_seq_sim, ClonalTree

CASES = [
    (
        2024, (10, 0.05, 0.05, 100, 20.0),
        [0.44000000000000006, 0.26, 10.000000000000117, 5.000000000000003, 1.9664902998235978,
         0.7142857142857064, 0.0, 0.0, 0.044000000000000004, 0.052000000000000005,
         1.0000000000000118, 1.0000000000000004, 0.19664902998235978, 0.1428571428571413, 0.0,
         0.0, 0.0, 0.06, 2.120914574274092, 1.8, 1.6711111111111114, -0.6259665374384215, 0.2,
         0.3333333333333333, 0.0, 0.1787184009406203, -0.23954238362052505, 0.5591157914436244,
         0.21769298413782748, 100.0],
    ),
    (
        7, (25, 0.01, 0.03, 300, 50.0),
        [10.667199999999989, 3.633600000000001, 323.0489417989486, 110.00000000000206,
         42.989294510733146, 16.660805941050516, 3.0, 0.0, 0.03282215384615381,
         0.03303272727272728, 0.9939967439967649, 1.0000000000000187, 0.13227475234071737,
         0.15146187219136834, 0.009230769230769232, 0.0, 0.11764705882352944, 0.1,
         7.9450032515583615, 7.033333333333333, 17.77888888888889, -0.42913482496919536,
         0.13793103448275862, 0.2, 2.0, 0.13712666770524978, 0.03456325945245924,
         16.255696062095346, 0.1482122553131084, 300.0],
    ),
]


@pytest.mark.parametrize("seed, params, expected", CASES)
def test_matches_original_implementation(seed, params, expected):
    n, rho, theta, L, delta = params
    np.random.seed(seed)
    tree = ClonalTree(n)
    s = ClonalOrigin_seq_sim(tree, rho, theta, L, delta)
    expected = np.asarray(expected)
    keep = np.r_[0:26, 29]
    np.testing.assert_allclose(s[keep], expected[keep], rtol=1e-12, atol=0)
    np.testing.assert_allclose(s[26:29], expected[26:29], rtol=1e-6, atol=1e-9)
