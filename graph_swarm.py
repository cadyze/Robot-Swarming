import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation

def graph_arena(grid_size, target_coords, robot_history, obstacles, tracked_robot=-1):
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
        robot_positions = [[] for _ in range(len(robot_history))]
        prev_robot_positions = [[] for _ in range(len(robot_history))]
        for hist in range(len(robot_history)):
            robot_positions[hist] = robot_history[hist][frame]
            if frame != 0:
                prev_robot_positions[hist] = robot_history[hist][frame - 1]
        tracked_robot_node = (0, 0)
        node_color = []
        for node in G.nodes():
            if node in robot_positions:
                index = robot_positions.index(node)
                if index == tracked_robot:
                    tracked_robot_node = node
                    node_color.append('green')
                else:
                    node_color.append('red')
            else:
                node_color.append('black')

        nx.draw(G, pos, with_labels=False, node_size=((figure_size[0] + 500) / grid_size), node_color=node_color, edge_color='gray', ax=ax)

        # Draw the lines between current and previous position
        for i in range(len(robot_positions)):
            if prev_robot_positions[i] != []:
                prev_pos = prev_robot_positions[i]
                curr_pos = robot_positions[i]
                x_line = [pos[prev_pos][0], pos[curr_pos][0]]
                y_line = [pos[prev_pos][1], pos[curr_pos][1]]
                if i == tracked_robot:
                    ax.plot(x_line, y_line, color='green')
                else:
                    ax.plot(x_line, y_line, color='red')

        
        # Draw the target area
        if graph_target_coords:
            x_coords, y_coords = zip(*graph_target_coords)
            ax.fill(x_coords, y_coords, color='red', alpha=0.5)
        
        # Fill the space in the polygon formed by obstacles
        if obstacles:
            for obstacle in obstacles:
                obstacle_coords = [pos[nodes] for nodes in obstacle]
                x_obs, y_obs = zip(*obstacle_coords)
                ax.fill(x_obs, y_obs, color='black', alpha=0.5)


    ani = animation.FuncAnimation(fig, update, frames=len(robot_history[0]), repeat=True, interval=500)

    plt.show()
