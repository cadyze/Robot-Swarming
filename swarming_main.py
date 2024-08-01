import matplotlib.pyplot as plt
import swarming_simulation

def plot_histogram(data, grid_size):
    plt.hist(data, bins='auto', edgecolor='black')
    plt.xlabel('Number of Moves')
    plt.ylabel('Frequency')
    plt.title('Number of Moves to Find Target in a {}x{} Arena'.format(grid_size, grid_size))
    plt.show()
    plt.s

border_obstacle = [ (7, 5), (7, 4), (7, 3), (6, 3), (5, 3), (4, 4), (3, 5), (3, 6), (3, 7), (4, 7), (5, 7), (5, 6), (4, 6), (4, 5), (5, 4), (6, 4), (6, 5)]
obstacle2 = [(4, 2), (3, 3), (4, 3), (5, 2)]
moves = []
grid_size = 7
collision_protocol = swarming_simulation.COLLISION_PROTOCOL.BREAK
obstacles = []
# Arena: 31x31, 151x151, 301x301
# Number of Robots: 1, 5, 10, 20
# Protocols: BREAK,  FIND NEXT AVAILABLE
# Create the SwarmingSimulation
RobotSwarmSimulator = swarming_simulation.SwarmSimulator(grid_size, (grid_size // 2, grid_size // 2), obstacles)
for i in range(1):
    moves.append(RobotSwarmSimulator.start_robot_swarming(2, collision_protocol, show_graph=True, wait_for_all=True))
plot_histogram(moves, grid_size)
