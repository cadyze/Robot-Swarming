import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation

def graph_arena(grid_size, target_coords, robot_history):
    grid_size -= 1
    # Given it's a square arena, we can slightly shear grid to become arena
    arena_size = (grid_size * 2, grid_size) # Two equalateral triangles per square unit after augments
    G = nx.triangular_lattice_graph(arena_size[1], arena_size[0])

    pos = {}

    for node in G:
        x, y = node

        # Calculates the arena to shift and create equilateral triangles
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

    target_area = []
    target_area.append((target_coords[0] - 1, target_coords[1]))
    target_area.append((target_coords[0] - 1, target_coords[1] + 1))
    target_area.append((target_coords[0], target_coords[1] + 1))
    target_area.append((target_coords[0] + 1, target_coords[1]))
    target_area.append((target_coords[0] + 1, target_coords[1] - 1))
    target_area.append((target_coords[0], target_coords[1] - 1))
    graph_target_coords = [pos[node] for node in target_area]
    
    figure_size = (min(arena_size[0], 20), min(arena_size[1], 10))
    fig, ax = plt.subplots(figsize=figure_size)

    def update(frame):
        ax.clear()
        current_target = robot_history[frame]
        node_color = ['red' if node == current_target else 'black' for node in G.nodes()]
        nx.draw(G, pos, with_labels=False, node_size=((figure_size[0] + 500) / grid_size), node_color=node_color, edge_color='gray', ax=ax)
        
        # Draw the target area
        if graph_target_coords:
            x_coords, y_coords = zip(*graph_target_coords)
            ax.fill(x_coords, y_coords, color='red', alpha=0.5)

    ani = animation.FuncAnimation(fig, update, frames=len(robot_history), repeat=True, interval=500)

    plt.show()
