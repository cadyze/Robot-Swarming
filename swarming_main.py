import matplotlib.pyplot as plt
import swarming_simulation

def plot_histogram(data):
    plt.hist(data, bins='auto', edgecolor='black')
    plt.xlabel('Number of Moves')
    plt.ylabel('Frequency')
    plt.title('Number of Moves to Find Target')
    plt.show()

# border_obstacle = [ (7, 5), (7, 4), (7, 3), (6, 3), (5, 3), (4, 4), (3, 5), (3, 6), (3, 7), (4, 7), (5, 7), (5, 6), (4, 6), (4, 5), (5, 4), (6, 4), (6, 5)]
moves = []
for i in range(100):
    moves.append(swarming_simulation.start_robot_swarming(10, (5, 5), 5, obstacles=[], show_graph=False))
plot_histogram(moves)
