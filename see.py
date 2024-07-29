import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def graph_arena(grid_size, target_coords_list):
    grid_size -= 1
    arena_size = (grid_size * 2, grid_size)  # Two equilateral triangles per square unit after augments
    G = nx.triangular_lattice_graph(arena_size[1], arena_size[0])

    pos = {}

    for node in G:
        x, y = node
        pos[node] = (x + (0.5 * y), y)
    
    edges = list(G.edges())
    to_add = []

    for edge in edges:
        u, v = edge
        if (v[0] - u[0], v[1] - u[1]) == (1, 1):
            to_add.append(((v[0] - 1, v[1]), (u[0] + 1, u[1])))
            G.remove_edge(u, v)
    
    for addition in to_add:
        G.add_edge(addition[0], addition[1])

    figure_size = (min(arena_size[0], 20), min(arena_size[1], 10))
    fig, ax = plt.subplots(figsize=figure_size)

    def update(frame):
        ax.clear()
        current_target = target_coords_list[frame]
        node_color = ['red' if node == current_target else 'black' for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_size=((figure_size[0] + 500) / grid_size), node_color=node_color, edge_color='gray', ax=ax)

    ani = animation.FuncAnimation(fig, update, frames=len(target_coords_list), repeat=True, interval=500)
    
    plt.show()

# Example usage
target_coords_list = [(0, 0), (2, 0), (0, 1), (1, 1), (2, 1)]  # List of target coordinates to animate
graph_arena(10, target_coords_list)
