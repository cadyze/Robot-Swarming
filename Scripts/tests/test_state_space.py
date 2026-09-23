import pytest

from swarm_sim.state_space import info_to_state, state_to_info


@pytest.mark.parametrize("grid_size", [4, 5, 11])
def test_round_trip(grid_size):
    for x in range(grid_size):
        for y in range(grid_size):
            for orientation in range(6):
                state = info_to_state(x, y, orientation, grid_size)
                x2, y2, o2 = state_to_info(state, grid_size)
                assert (x2, y2, o2) == (x, y, orientation)


def test_known_values():
    # state = 6 * (x + grid_size * y) + orientation
    assert info_to_state(0, 0, 0, grid_size=5) == 0
    assert info_to_state(1, 0, 0, grid_size=5) == 6
    assert info_to_state(0, 1, 0, grid_size=5) == 30
    assert info_to_state(2, 3, 4, grid_size=5) == 6 * (2 + 5 * 3) + 4
