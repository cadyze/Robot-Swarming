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
        return HTmu[0], HTstd[0]
