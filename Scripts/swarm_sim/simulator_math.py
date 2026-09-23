from swarm_sim.enums import STARTING_POSITION
from swarm_sim.swarm_simulator import SwarmSimulator


class SimulatorMath:
    @staticmethod
    def get_simulation_moments(grid_size, laziness_probablity):
        mathMatrix = SwarmSimulator(
            grid_size, (grid_size // 2, grid_size // 2), [],
            STARTING_POSITION.FILL, sensing_range=1, laziness_prob=laziness_probablity,
            for_math=True)
        HTmu, HTvariance, HTstd = mathMatrix.calculate_mean_variance()
        return HTmu[0], HTstd[0]
