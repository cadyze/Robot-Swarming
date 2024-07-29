import sys
import numpy as np
import scipy as sp
import scipy.sparse.linalg as lg
from scipy.sparse import lil_matrix
import random 
import statistics as stat
import statistics
import matplotlib.pyplot as plt
from scipy import stats

gridsize=4 # lil matrix good enough for 3000 grid size

'''
Given a 3x3 matrix, it will rotate the elements counter-clockwise while maintaining diagonal
'''
def rotate_matrix(matrix):
    u_matrix = np.matrix(matrix)[::-1]
    temp_matrix = np.matrix(matrix)[::-1]
    u_matrix[0, 1] = temp_matrix[1, 2]
    u_matrix[1, 2] = temp_matrix[2, 2]
    u_matrix[2, 2] = temp_matrix[2, 1]
    u_matrix[2, 1] = temp_matrix[1, 0]
    u_matrix[1, 0] = temp_matrix[0, 0]
    u_matrix[0, 0] = temp_matrix[0, 1]
    return u_matrix[::-1]

'''
Properly orientates the matrix to visually compare to what is graphed
'''
def print_orientated_matrix(matrix):
    print("\nORIENTATED MATRIX: \n {} \n\n".format(matrix[::-1]))
'''
Append a 3x3 matrix to the matrix
'''
def append_section_to_matrix(matrix, coords, section):
    x, y = coords
    # print(coords)
    # print(section)
    x_trim = (max(0, x-1), min(len(matrix) - 1, x+1))
    y_trim = (max(0, y-1), min(len(matrix[x]) - 1, y+1))
    # print(x_trim, y_trim)
    # print_orientated_matrix(matrix[x, y][y_trim[0] : y_trim[1] + 1, x_trim[0] : x_trim[1] + 1])
    matrix[x, y][y_trim[0] : y_trim[1] + 1, x_trim[0] : x_trim[1] + 1] = section
    # print_orientated_matrix(matrix[x, y][y_trim[0] : y_trim[1] + 1, x_trim[0] : x_trim[1] + 1])

def flip_section(section):
    return np.fliplr(np.flipud(section))

def init_arena(arena_size, target_coords):
    def print_arenas(debug_coords):
        print("PRINTING FOR COORDS: {}".format(debug_coords))
        print("\nOrientated 0 Arena: \n{}".format(orientated_0[debug_coords][::-1]))
        print("\nOrientated 60 Arena: \n{}".format(orientated_60[debug_coords][::-1]))
        print("\nOrientated 120 Arena: \n{}".format(orientated_120[debug_coords][::-1]))
        print("\nOrientated 180 Arena: \n{}".format(orientated_180[debug_coords][::-1]))
        print("\nOrientated 240 Arena: \n{}".format(orientated_240[debug_coords][::-1]))
        print("\nOrientated 300 Arena: \n{}".format(orientated_300[debug_coords][::-1]))
        print("\n----------------------------------\n\n")
    
    # Six different matrices, each representing the probability matrix for each orientation (measured in degrees)
    I_ACTIONS = {'FORWARD': 0.49, 'BACKWARD': 0.01, 'SMALL_TURN': 0.24, 'BIG_TURN': 0.01}	# I: Inside
    CW_ACTIONS = {'REFLECT': 0.6, 'BACKWARD': 0.15, 'SMALL_TURN': 0.15, 'BIG_TURN': 0.1}	# CW: Collide against the wall
    PW_ACTIONS = {'FORWARD': 0.45, 'BACKWARD': 0.04, 'SMALL_TURN': 0.45, 'BIG_TURN': 0.06}	# PW: Parallel with the wall
    ASC_ACTIONS = {'BACKWARD': 0.5, 'BIG_TURN': 0.5}										# ASC: Along the sharp corner
    AOC_ACTIONS = {'BACKWARD': 0.25, 'SMALL_TURN': 0.25, 'BIG_TURN': 0.5}					# AOC: Along the obtuse corner
    COC_ACTIONS = {'BACKWARD': 0.34, 'BIG_TURN': 0.33}										# COC: Collide toward the obtuse corner
    
    # This square matrix encapsulates all possible actions for a given point w/ center at (1, 1)
    # Below they are all initialized with orientation facing 0 and will be rotated around
    I_PROB_0 = np.matrix([[I_ACTIONS['BIG_TURN'],     I_ACTIONS['SMALL_TURN'],    0],
                        [I_ACTIONS['BACKWARD'],     0,                          I_ACTIONS['FORWARD']],
                        [0,                         I_ACTIONS['BIG_TURN'],      I_ACTIONS['SMALL_TURN']]][::-1])
    
    I_PROB_DICT = {}
    next = I_PROB_0
    for orientation in [0, 60, 120, 180, 240, 300]:
        I_PROB_DICT[orientation] = next
        next = rotate_matrix(next)

    
    CW_PROB_DICT = {0: np.matrix([[CW_ACTIONS['REFLECT'],   CW_ACTIONS['SMALL_TURN'], 0],
                        [CW_ACTIONS['BACKWARD'],     0,                          0],
                        [0,                         CW_ACTIONS['BIG_TURN'],     0]][::-1]),
                        
                    60: np.matrix([[CW_ACTIONS['BIG_TURN'],     0,                          CW_ACTIONS['SMALL_TURN']],
                        [0,                         CW_ACTIONS['REFLECT'],     CW_ACTIONS['BACKWARD']]][::-1]),
                        
                    120: np.matrix([[0,                 CW_ACTIONS['SMALL_TURN'],   0], 
                        [CW_ACTIONS['SMALL_TURN'],      0,                          CW_ACTIONS['BIG_TURN']],
                        [0,                             CW_ACTIONS['BACKWARD'],     CW_ACTIONS['REFLECT']]][::-1]),

                    180: np.matrix([[0,   CW_ACTIONS['BIG_TURN'], 0],
                        [0,     0,                          CW_ACTIONS['BACKWARD']],
                        [0,                         CW_ACTIONS['SMALL_TURN'],     CW_ACTIONS['REFLECT']]][::-1]),

                    240: np.matrix([[CW_ACTIONS['REFLECT'],   CW_ACTIONS['BACKWARD'], 0],
                        [CW_ACTIONS['SMALL_TURN'],     0,                          CW_ACTIONS['BIG_TURN']]][::-1]),
                        
                    300: np.matrix([[CW_ACTIONS['BACKWARD'],   CW_ACTIONS['REFLECT'], 0],
                        [CW_ACTIONS['BIG_TURN'],     0,                          CW_ACTIONS['SMALL_TURN']],
                        [0,                         CW_ACTIONS['SMALL_TURN'],    0]][::-1])
                        
                    }
    # Note: These states can never occur with 120 or 300
    C_PW_PROB_0 = np.matrix([[0,                      0,                          0],
                        [PW_ACTIONS['BACKWARD'],    0,                          PW_ACTIONS['FORWARD']],
                        [0,                         PW_ACTIONS['BIG_TURN'],     PW_ACTIONS['SMALL_TURN']]][::-1])
    CC_PW_PROB_0 = np.matrix([[PW_ACTIONS['BIG_TURN'],      PW_ACTIONS['SMALL_TURN'],       0],
                        [PW_ACTIONS['BACKWARD'],    0,      PW_ACTIONS['FORWARD']],
                        [0,                         0,      0]][::-1])
    
    # Note: Because it's the same probability either way, there's only one possible matrix
    ASC_PROB = np.matrix([[ASC_ACTIONS['BACKWARD'],   	0,                     ],
                        [0,                         	ASC_ACTIONS['BIG_TURN']]][::-1])
    AOC_PROB_BOTR = np.matrix([[AOC_ACTIONS['BIG_TURN'], AOC_ACTIONS['SMALL_TURN']],
                        [AOC_ACTIONS['BACKWARD'],   	 0,                       ]][::-1])
    AOC_PROB_TOPL = np.matrix([[0, 							AOC_ACTIONS['SMALL_TURN']],
                        [AOC_ACTIONS['BACKWARD'],   	 	AOC_ACTIONS['BIG_TURN']]][::-1])
    
    # Note: These account for following the border, but not in the reverse order
    # PW follows for when going clockwise, will flip later
    C_PW_PROB_DICT, CC_PW_PROB_DICT = {}, {}
    next = [C_PW_PROB_0, CC_PW_PROB_0] 
    for orientation in [0, 60, 120, 180, 240, 300]:
        if not (orientation == 120 or orientation == 300):
            C_PW_PROB_DICT[orientation] = next[0] # Note: will be creating the matrix for rotating later
            CC_PW_PROB_DICT[orientation] = next[1] # Note: will be creating the matrix for rotating later
        next = [rotate_matrix(next[0]), rotate_matrix(next[1])]

    # Note: This state is only possible for orientations of 300 and 120, the one below is for 300
    COC_PROB_300 = np.matrix([[COC_ACTIONS['BACKWARD'], COC_ACTIONS['BIG_TURN']],
                        [COC_ACTIONS['BIG_TURN'],   0                         ]][::-1])

    # Now that the matrices are created, we can establish the probability matrix for every orientation

    #  How the following matrices are setup:
    #  (CURRENT_X, CURRENT_Y, NEXT_X, NEXT_Y) -> Where NEXT represents the point that the current point may transition to.
    #  eg. (0, 1, 2, 3) -> returns the probability the robot on point (0, 1) transitions to (2, 3) given the orientation
    orientated_0 = np.zeros((arena_size, arena_size, arena_size, arena_size))
    orientated_60 = np.zeros((arena_size, arena_size, arena_size, arena_size))
    orientated_120 = np.zeros((arena_size, arena_size, arena_size, arena_size))
    orientated_180 = np.zeros((arena_size, arena_size, arena_size, arena_size))
    orientated_240 = np.zeros((arena_size, arena_size, arena_size, arena_size))
    orientated_300 = np.zeros((arena_size, arena_size, arena_size, arena_size))

    # For every single orientation, the inside boundaries and probabilities are the same
    for x in range(1, arena_size - 1):
        for y in range(1, arena_size - 1):
            coords = (x, y)
            append_section_to_matrix(orientated_0, coords, I_PROB_DICT[0])
            append_section_to_matrix(orientated_60, coords, I_PROB_DICT[60])
            append_section_to_matrix(orientated_120, coords, I_PROB_DICT[120])
            append_section_to_matrix(orientated_180, coords, I_PROB_DICT[180])
            append_section_to_matrix(orientated_240, coords, I_PROB_DICT[240])
            append_section_to_matrix(orientated_300, coords, I_PROB_DICT[300])
    
    # print_orientated_matrix(orientated_0[3, 2])
    # Adjusting the top edge excluding the corners
    y = arena_size - 1 # This is top y-level
    ADJUSTED_PW_180 = np.fliplr(np.flipud(CC_PW_PROB_DICT[180][1:, :]))
    for x in range(1, arena_size - 1):
        coords = (x, y)
        # print("\nMATRIX TO APPEND: \n{}\n".format(CW_PROB_DICT[120][:2]))
        append_section_to_matrix(orientated_0, coords, C_PW_PROB_DICT[0][:2, :]) # Trims the given matrix to fit
        append_section_to_matrix(orientated_180, coords, ADJUSTED_PW_180) # Flips to account for counter-clockwise
     
        append_section_to_matrix(orientated_60, coords, CW_PROB_DICT[60]) # Trims the given matrix to fit
        append_section_to_matrix(orientated_120, coords, CW_PROB_DICT[120][:2]) # Trims the given matrix to fit
    

    # Adjusting the bottom edge excluding the corners
    y = 0
    for x in range(1, arena_size - 1):
        coords = (x, y)
        # print_orientated_matrix(CC_PW_PROB_DICT[0][1:])
        # print("\nMATRIX TO APPEND: \n{}\n".format(C_PW_PROB_DICT[180][:, :][::-1]))
        # print("\nMATRIX TO APPEND: \n{}\n".format(CC_PW_PROB_DICT[180][:, :][::-1]))
        append_section_to_matrix(orientated_0, coords, CC_PW_PROB_DICT[0][1:]) # Trims the given matrix to fit
        append_section_to_matrix(orientated_180, coords, C_PW_PROB_DICT[180][1:, :]) # Flips to account for counter-clockwise
     
        append_section_to_matrix(orientated_240, coords, CW_PROB_DICT[240]) # Trims the given matrix to fit
        append_section_to_matrix(orientated_300, coords, CW_PROB_DICT[300][1:]) # Trims the given matrix to fit
    # print_arenas((1, 0))

    # Adjusting the left edge excluding the corners
    x = 0
    for y in range(1, arena_size - 1):
        coords = (x, y)
        # print_orientated_matrix(CW_PROB_DICT[120][:, 1:])
        append_section_to_matrix(orientated_180, coords, CW_PROB_DICT[180][:, 1:]) # Trims the given matrix to fit
        append_section_to_matrix(orientated_120, coords, CW_PROB_DICT[120][:, 1:])

        append_section_to_matrix(orientated_60, coords, C_PW_PROB_DICT[60][:, 1:])
        append_section_to_matrix(orientated_240, coords, CC_PW_PROB_DICT[240][:, 1:])
    # print_arenas((9, 2))

    # Adjusting the right edge excluding the corners
    x = arena_size - 1
    for y in range(1, arena_size - 1):
        coords = (x, y)
        # print_orientated_matrix(C_PW_PROB_DICT[240][:, :])
        append_section_to_matrix(orientated_0, coords, CW_PROB_DICT[0][:, :2]) # Trims the given matrix to fit
        append_section_to_matrix(orientated_300, coords, CW_PROB_DICT[300][:, :2])

        append_section_to_matrix(orientated_60, coords, CC_PW_PROB_DICT[60][:, :2])
        append_section_to_matrix(orientated_240, coords, C_PW_PROB_DICT[240][:, :2])
    # print_arenas((9, 1))

    # Bottom-left Corner
    coords = (0, 0)
    append_section_to_matrix(orientated_240, coords, ASC_PROB)
    append_section_to_matrix(orientated_180, coords, ASC_PROB)
    # print_arenas(coords)

    # Bottom-right corner
    coords = (arena_size - 1, 0)
    # print_orientated_matrix(COC_PROB_300)
    append_section_to_matrix(orientated_0, coords, AOC_PROB_BOTR)
    append_section_to_matrix(orientated_240, coords, AOC_PROB_BOTR)
    append_section_to_matrix(orientated_300, coords, COC_PROB_300)
    # print_arenas(coords)

    # Top-left corner
    coords = (0, arena_size - 1)
    append_section_to_matrix(orientated_60, coords, AOC_PROB_TOPL)
    append_section_to_matrix(orientated_120, coords, np.fliplr(np.flipud(COC_PROB_300)))
    append_section_to_matrix(orientated_180, coords, AOC_PROB_TOPL)
    # print_arenas(coords)

    # Top-right corner
    coords = (arena_size - 1, arena_size - 1)
    append_section_to_matrix(orientated_0, coords, ASC_PROB)
    append_section_to_matrix(orientated_60, coords, ASC_PROB)
    # print_arenas(coords)
    # print_arenas((2, 0))
    '''
    For all orientations, guarantee a jump from the start pos to target pos
    '''
    def guarantee_hop(start_pos, target_pos):
        # print("GUARANTEEEEEE: FROM {} TO {}".format(start_pos, target_pos))
        orientated_0[start_pos[0], start_pos[1]] = 0
        orientated_0[start_pos[0], start_pos[1], target_pos[0], target_pos[1]] = 1

        orientated_60[start_pos[0], start_pos[1]] = 0
        orientated_60[start_pos[0], start_pos[1], target_pos[0], target_pos[1]] = 1

        orientated_120[start_pos[0], start_pos[1]] = 0
        orientated_120[start_pos[0], start_pos[1], target_pos[0], target_pos[1]] = 1
         
        orientated_180[start_pos[0], start_pos[1]] = 0
        orientated_180[start_pos[0], start_pos[1], target_pos[0], target_pos[1]] = 1
        
        orientated_240[start_pos[0], start_pos[1]] = 0
        orientated_240[start_pos[0], start_pos[1], target_pos[0], target_pos[1]] = 1
        
        orientated_300[start_pos[0], start_pos[1]] = 0
        orientated_300[start_pos[0], start_pos[1], target_pos[0], target_pos[1]] = 1
        return

    # Create absorbing nodes surrounding the target
    x, y = target_coords
    for i in range(-1, 2):
        for j in range(-1, 2):
            # Middle row has three absorbing spots
            if i == -1 and j == -1 or i == 1 and j == 1:
                continue
            guarantee_hop((x+i, y+j), target_coords)

    # Turn all of the matrices into a transition matrix
    orientation_state = 0
    transition_matrix = np.zeros((6, arena_size**2, arena_size**2))	
    for orientation_matrix in [orientated_0, orientated_60, orientated_120, orientated_180, orientated_240, orientated_300]:
        # Note to access any element from the 4d array, the equation is: (i, j, k, l) == (i*n+j, k*n+l)
        transition_matrix[orientation_state] = orientation_matrix.reshape(arena_size**2, arena_size**2)
        # (0, 0) -> (1, 0) -> E (0 degrees) 
        print(np.shape(transition_matrix))
        orientation_state += 1
    # print(np.shape(transition_matrix))
    return transition_matrix
    
def get_robot_next_move(robot_state, transition_matrix, arena_size):
    # Get next move based on the transition_matrix
    # print(robot_state)
    current_pos = robot_state[1]
    # print("CURRENT POS {}".format(current_pos))
    probabilities = transition_matrix[robot_state[0], current_pos[0] * arena_size + current_pos[1]]
    # print(probabilities)
    next_pos = np.random.choice(len(probabilities), p=probabilities)
    new_pos = (next_pos % arena_size, next_pos // arena_size)	# Convert 1D return to 2D for analysis

    # Using the new coordinates, update the orientation of the robot
    y_diff = new_pos[1] - current_pos[1]
    x_diff = new_pos[0] - current_pos[0]
    orientation_state = 0

    if x_diff == 1 and y_diff == 0:		# 0 
        orientation_state = 0
    elif x_diff == 0 and y_diff == 1:	# 60
        orientation_state = 1
    elif x_diff == -1 and y_diff == 1:	# 120
        orientation_state = 2
    elif x_diff == -1 and y_diff == 0:	# 180
        orientation_state = 3
    elif x_diff == 0 and y_diff == -1:	# 240
        orientation_state = 4
    else:                               # 300
        orientation_state = 5

    return [orientation_state, new_pos]

# def arena_state_to_transition_matrix():


def run_robot_swarming(arena_size, target_coords):
    probability_matrix = init_arena(arena_size, target_coords)
    robot_state = [0, (1, 0)] # (i, j, k) | i = orientation, (j, k) = robot position

    #TODO: Create a 600x600 (when with arena size 10) transitional matrix getter
        # (s1, s2) -> both s1, s2 represent a state -> a state contains [0, (0, 1)] following above format
        # transitional_matrix[s1, s2] -> 0.6 where s1 = [0, 0, 0] | s2 = [0, 1, 0]
    
    print(np.shape(probability_matrix))
    for o in probability_matrix:
        print(o)

    
    robot_move_history = []
    curr_pos = robot_state[1]
    print("There is a robot at {} and the target is at {}.\n Simulation Begins now.".format(curr_pos, target_coords))
    while(curr_pos != target_coords):
        robot_move_history.append(get_robot_next_move(robot_state, probability_matrix, arena_size))
        robot_state = robot_move_history[-1]
        curr_pos = robot_state[1]
        print(robot_state)
    print("Converged to the target in {} moves.".format(len(robot_move_history)))
    graph_arena(arena_size, target_coords, robot_move_history)
    return

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

    target_area = []
    target_area.append((target_coords[0] - 1, target_coords[1]))
    target_area.append((target_coords[0] - 1, target_coords[1] + 1))
    target_area.append((target_coords[0], target_coords[1] + 1))
    target_area.append((target_coords[0] + 1, target_coords[1]))
    target_area.append((target_coords[0] + 1, target_coords[1] - 1))
    target_area.append((target_coords[0], target_coords[1] - 1))
    graph_target_coords = [pos[node] for node in target_area]
    
    figure_size = (min(arena_size[0], 20), min(arena_size[1], 10))
    plt.figure(figsize=figure_size)
    nx.draw(G, pos, with_labels=True, node_size=((figure_size[0] + 500) / grid_size), node_color='black', edge_color='gray')

    figure_size = (min(arena_size[0], 20), min(arena_size[1], 10))
    fig, ax = plt.subplots(figsize=figure_size)
    robot_history = [hist[1] for hist in robot_history]
    print(robot_history)

    def update(frame):
        ax.clear()
        current_target = robot_history[frame]
        node_color = ['red' if node == current_target else 'black' for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_size=((figure_size[0] + 500) / grid_size), node_color=node_color, edge_color='gray', ax=ax)

    ani = animation.FuncAnimation(fig, update, frames=len(robot_history), repeat=True, interval=500)
    
    # Draw the absorbing nodes and the target
    if graph_target_coords:
        x_coords, y_coords = zip(*graph_target_coords)
        plt.fill(x_coords, y_coords, color='red', alpha=0.5)

    plt.show()

run_robot_swarming(10, (5, 5))