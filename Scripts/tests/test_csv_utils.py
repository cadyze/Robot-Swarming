import numpy as np
import pandas as pd
import pytest

from swarm_sim.csv_utils import CSVUtils


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "Timesteps": [5, 8, 50, 12, 9, 200, 15, 11],
        "Collisions": [0, 1, 2, 0, 1, 3, 0, 1],
        "Steps Waited": [1, 2, 5, 1, 2, 10, 1, 2],
        "Real-Time Elapsed": [0.1] * 8,
    })
    df.to_csv(path, index=False)
    return str(path)


def test_filters_timesteps_at_or_below_ten(sample_csv):
    utils = CSVUtils(sample_csv)
    # Values <= 10 are dropped (5, 8, 9) leaving [50, 12, 200, 15, 11]
    assert sorted(utils.m_values.tolist()) == [11, 12, 15, 50, 200]


def test_mean_and_variance(sample_csv):
    utils = CSVUtils(sample_csv)
    expected = np.array([50, 12, 9, 200, 15, 11])
    expected = expected[expected > 10]
    assert utils.get_mean() == pytest.approx(expected.mean())
    assert utils.get_variance() == pytest.approx(expected.std())
    assert utils.get_count() == len(expected)


def test_sample_falls_back_to_full_population_when_too_small(sample_csv):
    utils = CSVUtils(sample_csv)
    utils.generate_new_sample(1000)  # bigger than the population
    assert utils.get_sampled_mean() == pytest.approx(utils.get_mean())
    assert utils.get_sampled_variance() == pytest.approx(utils.get_variance())


def test_wait_percent(sample_csv):
    utils = CSVUtils(sample_csv)
    df = pd.read_csv(sample_csv)
    expected = df["Steps Waited"].sum() / df["Timesteps"].sum() * 100
    assert utils.get_wait_percent() == pytest.approx(expected)
