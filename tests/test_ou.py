import numpy as np
import pytest

from decoy.ou_process import OUProcess


def test_stationary_distribution():
    theta = 0.5
    mu = 2.0
    sigma = 1.0
    dt = 0.001

    n_steps = 50_000
    burn_in = 20_000
    n_trajectories = 20

    rng = np.random.default_rng(42)

    stationary_samples = []

    for _ in range(n_trajectories):
        process = OUProcess(
            theta=theta,
            mu=mu,
            sigma=sigma,
            dt=dt,
            rng=rng,
        )

        trajectory = process.simulate(
            x0=10.0,
            n_steps=n_steps,
        )

        stationary_samples.append(trajectory[burn_in:])

    samples = np.concatenate(stationary_samples)

    empirical_mean = np.mean(samples)
    empirical_variance = np.var(samples)

    theoretical_mean = mu
    theoretical_variance = sigma**2 / (2 * theta)

    assert np.isclose(
        empirical_mean,
        theoretical_mean,
        atol=0.05,
    )

    assert np.isclose(
        empirical_variance,
        theoretical_variance,
        rtol=0.10,
    )


def test_mean_reversion():
    theta = 0.5
    mu = 0.0
    sigma = 0.0
    dt = 0.001

    rng = np.random.default_rng(42)

    process = OUProcess(
        theta=theta,
        mu=mu,
        sigma=sigma,
        dt=dt,
        rng=rng,
    )

    trajectory = process.simulate(
        x0=10.0,
        n_steps=10_000,
    )

    assert abs(trajectory[-1] - mu) < abs(trajectory[0] - mu)


@pytest.mark.parametrize(
    "theta, sigma, dt",
    [
        (0.0, 1.0, 0.01),
        (-0.1, 1.0, 0.01),
        (0.5, 1.0, 0.0),
        (0.5, 1.0, -0.01),
        (0.5, -1.0, 0.01),
    ],
)
def test_invalid_parameters(theta, sigma, dt):
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        OUProcess(
            theta=theta,
            mu=0.0,
            sigma=sigma,
            dt=dt,
            rng=rng,
        )