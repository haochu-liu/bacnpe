"""Tests for bacnpe.npe."""
import math
import warnings

import pytest
import torch

from bacnpe.npe import DiscreteUniform, LeaveLengthOut_NN, build_prior

RANGES = dict(rho_range=(0.0, 0.2), theta_range=(0.0, 0.1), L_range=(100, 10000))


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


# ----------------------------------------------------------- DiscreteUniform
def test_discrete_uniform_sample_and_log_prob():
    d = DiscreteUniform(torch.tensor([3.0]), torch.tensor([6.0]))
    s = d.sample((4000,))
    assert s.shape == (4000, 1) and s.dtype == torch.float32
    assert torch.equal(torch.unique(s), torch.tensor([3.0, 4.0, 5.0, 6.0]))   # both ends hit
    lp = d.log_prob(torch.tensor([[3.0], [4.5], [6.0], [2.0], [7.0]]))
    expected = -math.log(4)
    assert torch.allclose(lp[:3], torch.full((3, 1), expected))           # non-integer inside is fine
    assert torch.all(torch.isinf(lp[3:])) and torch.all(lp[3:] < 0)
    assert d.mean.item() == 4.5 and d.variance.item() == pytest.approx((16 - 1) / 12)
    assert d.support.check(torch.tensor([4.5])).item()


def test_discrete_uniform_batched_bounds():
    d = DiscreteUniform(torch.tensor([0.0, 100.0]), torch.tensor([1.0, 102.0]))
    s = d.sample((2000,))
    assert set(s[:, 0].unique().tolist()) == {0.0, 1.0}
    assert set(s[:, 1].unique().tolist()) == {100.0, 101.0, 102.0}


def test_discrete_uniform_bad_args():
    with pytest.raises(ValueError):
        DiscreteUniform(5.0, 4.0)
    with pytest.raises(ValueError):
        DiscreteUniform(0.5, 4.0)


# ----------------------------------------------------------- LeaveLengthOut_NN
def test_embedding_appends_length():
    torch.manual_seed(0)
    net = LeaveLengthOut_NN(input_dim=30, num_hiddens=16, num_hidden_layers=1, num_outputs=2)
    x = torch.randn(7, 30)
    out = net(x)
    assert out.shape == (7, 3)
    assert torch.equal(out[:, -1], x[:, -1])
    net2 = LeaveLengthOut_NN(input_dim=30, hidden_dims=[8, 4], num_outputs=5)
    assert net2(x).shape == (7, 6)
    assert [m.out_features for m in net2.mlp if isinstance(m, torch.nn.Linear)] == [8, 4, 5]


# ----------------------------------------------------------- build_prior
def test_build_prior():
    prior = build_prior(**RANGES)
    s = prior.sample((2000,))
    assert s.shape == (2000, 3)
    assert torch.all((s[:, 0] >= 0) & (s[:, 0] <= 0.2))
    assert torch.all((s[:, 1] >= 0) & (s[:, 1] <= 0.1))
    assert torch.all((s[:, 2] >= 100) & (s[:, 2] <= 10000))
    assert torch.equal(s[:, 2], torch.round(s[:, 2]))
    expected = -math.log(0.2) - math.log(0.1) - math.log(9901)
    assert torch.allclose(prior.log_prob(s[:5]), torch.full((5,), expected), atol=1e-4)
    outside = torch.tensor([[0.3, 0.05, 500.0], [0.1, 0.05, 50.0]])
    assert torch.all(torch.isinf(prior.log_prob(outside)))


@pytest.mark.parametrize("kwargs, error", [
    (dict(rho_range=(0.1, 0.1)), ValueError),        # continuous prior needs low < high
    (dict(theta_range=(-0.1, 0.1)), ValueError),
    (dict(rho_range=(0.2, 0.1)), ValueError),
    (dict(L_range=(100.0, 200)), TypeError),
    (dict(L_range=(0, 200)), ValueError),
    (dict(theta_range=0.1), TypeError),
])
def test_build_prior_bad_inputs(kwargs, error):
    with pytest.raises(error):
        build_prior(**{**RANGES, **kwargs})
