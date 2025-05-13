import RobotSwarmingSimulator
from RobotSwarmingSimulator import COLLISION_PROTOCOL, STARTING_POSITION
import pandas as pd

class SimulatorMath: 
    def __init__(self):
        pass
    
    @staticmethod
    def get_simulation_moments(grid_size, laziness_probablity):
        # Create the SwarmingSimulation
        mathMatrix = RobotSwarmingSimulator.SwarmSimulator(grid_size, (grid_size // 2, grid_size // 2), [], 
                                                                    STARTING_POSITION.FILL, sensing_range=1, laziness_prob=laziness_probablity,
                                                                    for_math=True) 
        HTmu, HTvariance, HTstd = mathMatrix.calculate_mean_variance()
        # print("PROB: {} | HTMU: {} | HTVAR {}".format(laziness_probablity, round(HTmu[0], 3), round(HTstd[0], 3)))
        return HTmu, HTstd
    
    @staticmethod
    def get_simulation_moment_at(grid_size, pos, orientation, laziness_probability):
        state = SimulatorMath.get_state_from_details(grid_size, pos, orientation)
        HTmu, HTstd = SimulatorMath.get_simulation_moments(grid_size, laziness_probability)
        return HTmu[state], HTstd[state]

    
    @staticmethod
    def get_state_from_details(grid_size, pos, orientation):
        x, y = pos
        return 6 * (x + grid_size * y) + orientation


mu, std = SimulatorMath.get_simulation_moment_at(31, (3, 3), 0, 0)
print("AVG: {} | STD: {}".format(mu, std))