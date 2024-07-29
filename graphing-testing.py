import networkx as nx
import matplotlib.pyplot as plt

def create_arena(grid_size=10):
    grid_size -= 1

    # Given it's a square arena, we can slightly shear grid to become arena
    arena_size = (grid_size * 2, grid_size) # Two equalateral triangles per square unit after augments
    G = nx.triangular_lattice_graph(arena_size[1], arena_size[0])

    pos = {}

    for node in G:
        x, y = node

        # Calculates the arena to shit and create equalateral triangles
        pos[node] = (x + (0.5 * y), y)
    
    # Remove the edges that are incorrectly graphed to each other
    edges = G.edges()
    to_add = []

    for edge in edges:
        u, v = edge
        if (v[0] - u[0], v[1] - u[1]) == (1, 1):
            to_add.append(((v[0] - 1, v[1]), (u[0] + 1, u[1])))
            G.remove_edge(u, v)
    
    # Adds to the graph edges to connect the edges correctly
    for addition in to_add:
        G.add_edge(addition[0], addition[1])

    figure_size = (min(arena_size[0], 20), min(arena_size[1], 10))
    plt.figure(figsize=figure_size)
    nx.draw(G, pos, with_labels=True, node_size=((figure_size[0] + 500) / grid_size), node_color='black', edge_color='gray')
    plt.show()


import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

def draw_animated_triangular_lattice(m, n, history, filename=None, filetype='gif'):
    # Create a triangular lattice graph
    G = nx.triangular_lattice_graph(m, n)

    # Define positions for nodes to form equilateral triangles
    pos = {}
    for node in G.nodes():
        x, y = node
        pos[node] = (x + 0.5 * y, y * (3**0.5 / 2))

    fig, ax = plt.subplots(figsize=(10, 8))

    def update(num):
        ax.clear()
        edges = G.edges()
        edge_colors = ['gray' if (u, v) not in history[:num] else 'red' for u, v in edges]
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', edge_color=edge_colors)
        ax.set_title(f"Move {num + 1}")

    ani = FuncAnimation(fig, update, frames=len(history), interval=500, blit=False)

    if filename:
        if filetype == 'gif':
            ani.save(filename, writer=PillowWriter(fps=2))
        elif filetype == 'mov':
            ani.save(filename, writer=FFMpegWriter(fps=2))

    plt.show()

# Define the history of changes as a list of edges
history = [
    ((0, 0), (1, 0)),
    ((1, 0), (2, 0)),
    ((2, 0), (3, 0)),
    ((3, 0), (4, 0)),
    ((0, 1), (1, 1)),
    ((1, 1), (2, 1)),
    ((2, 1), (3, 1)),
    ((3, 1), (4, 1))
]

# Call the function to visualize the animated graph and save it as a GIF
create_arena(10)
