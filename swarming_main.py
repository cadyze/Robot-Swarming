import matplotlib.pyplot as plt
import RobotSwarmingSimulator
import swarming_simulator

def plot_histogram(data, grid_size):
    plt.hist(data, bins='auto', edgecolor='black')
    plt.xlabel('Number of Moves')
    plt.ylabel('Frequency')
    plt.title('Number of Moves to Find Target in a {}x{} Arena'.format(grid_size, grid_size))
    plt.show()

border_obstacle = [ (7, 5), (7, 4), (7, 3), (6, 3), (5, 3), (4, 4), (3, 5), (3, 6), (3, 7), (4, 7), (5, 7), (5, 6), (4, 6), (4, 5), (5, 4), (6, 4), (6, 5)]
obstacle2 = [(4, 2), (3, 3), (4, 3), (5, 2)]
moves = []
grid_size = 61
collision_protocol = RobotSwarmingSimulator.COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE
obstacles = []

def run_simulation(num_robots, collision_protocol, tracked_robot):  
    # Arena: 31x31, 61x61, 81x81
    # Number of Robots: 1, 5, 10, 20
    # Protocols: BREAK,  FIND NEXT AVAILABLE
    # Tracking Robots: First index, Last index, Middle Robot
    # Create the SwarmingSimulation
    history = []
    RobotSwarmSimulator = RobotSwarmingSimulator.SwarmSimulator(grid_size, (grid_size // 2, grid_size // 2), obstacles)
    for i in range(10000):
        # moves.append(swarming_simulator.start_robot_swarming(grid_size, (grid_size // 2, grid_size // 2), 1, collision_protocol, 
        #                                                      show_graph=True))
        history.append(RobotSwarmSimulator.start_robot_swarming(num_robots, collision_protocol, show_graph=False, tracked_robot=tracked_robot))
    return history

moves.append(run_simulation(20, collision_protocol, 0))
moves.append(run_simulation(20, collision_protocol, 19))
moves.append(run_simulation(20, collision_protocol, 9))

for m_list in moves:
    plot_histogram(m_list, grid_size)
