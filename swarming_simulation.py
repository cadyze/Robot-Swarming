import numpy as np
import graph_swarm

def init_arena(grid_size, target_pos): 
    # Six different matrices, each representing the probability matrix for each orientation (measured in degrees)
    I_ACTIONS = {'FORWARD': 0.49, 'BACKWARD': 0.01, 'SMALL_TURN': 0.24, 'BIG_TURN': 0.01}	# I: Inside
    CW_ACTIONS = {'REFLECT': 0.6, 'BACKWARD': 0.15, 'SMALL_TURN': 0.15, 'BIG_TURN': 0.1}	# CW: Collide against the wall
    PW_ACTIONS = {'FORWARD': 0.45, 'BACKWARD': 0.04, 'SMALL_TURN': 0.45, 'BIG_TURN': 0.06}	# PW: Parallel with the wall
    ASC_ACTIONS = {'BACKWARD': 0.5, 'BIG_TURN': 0.5}										# ASC: Along the sharp corner
    AOC_ACTIONS = {'BACKWARD': 0.25, 'SMALL_TURN': 0.25, 'BIG_TURN': 0.5}					# AOC: Along the obtuse corner
    COC_ACTIONS = {'BACKWARD': 0.34, 'BIG_TURN': 0.33}										# COC: Collide toward the obtuse corner

    '''
    For the following, this is how the arena is set up:
    It is a 2D transitional matrix that hops from state to state
    Each state is formatted as the following when given (x, y) = pos and o = orientation: 
        state = 6 * (x + grid_size * y) + orientation
    Defining states for o:
        o = 0 = 0   degrees
        o = 1 = 60  degrees
        0 = 2 = 120 degrees
        0 = 3 = 180 degrees
        0 = 4 = 240 degrees
        0 = 5 = 300 degrees
    '''
    arena_size = 6 * grid_size**2
    Arena = np.zeros((arena_size, arena_size))
    
    def info_to_state(x, y, orientation):
        return 6 * (x + grid_size * y) + orientation
    
    ORIENTATIONS = {
        "0" : 0,
        "60" : 1,
        "120" : 2,
        "180" : 3,
        "240" : 4,
        "300" : 5
    }

    # Modifying the insides
    for x in range(1, grid_size - 1):
        for y in range(1, grid_size - 1):
            # Modifying the orientation for 0 degrees
            Arena[info_to_state(x, y, ORIENTATIONS['0'])][info_to_state(x+1, y, ORIENTATIONS['0'])] = I_ACTIONS['FORWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['0'])][info_to_state(x, y+1, ORIENTATIONS['60'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['0'])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['0'])][info_to_state(x-1, y, ORIENTATIONS['180'])] = I_ACTIONS['BACKWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['0'])][info_to_state(x, y-1, ORIENTATIONS['240'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['0'])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = I_ACTIONS['SMALL_TURN']

            # 60 Degrees
            Arena[info_to_state(x, y, ORIENTATIONS['60'])][info_to_state(x+1, y, ORIENTATIONS['0'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['60'])][info_to_state(x, y+1, ORIENTATIONS['60'])] = I_ACTIONS['FORWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['60'])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['60'])][info_to_state(x-1, y, ORIENTATIONS['180'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['60'])][info_to_state(x, y-1, ORIENTATIONS['240'])] = I_ACTIONS['BACKWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['60'])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = I_ACTIONS['BIG_TURN']

            # 120 Degrees
            Arena[info_to_state(x, y, ORIENTATIONS['120'])][info_to_state(x+1, y, ORIENTATIONS['0'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['120'])][info_to_state(x, y+1, ORIENTATIONS['60'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['120'])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['FORWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['120'])][info_to_state(x-1, y, ORIENTATIONS['180'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['120'])][info_to_state(x, y-1, ORIENTATIONS['240'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['120'])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = I_ACTIONS['BACKWARD']

            # 180 Degrees
            Arena[info_to_state(x, y, ORIENTATIONS['180'])][info_to_state(x+1, y, ORIENTATIONS['0'])] = I_ACTIONS['BACKWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['180'])][info_to_state(x, y+1, ORIENTATIONS['60'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['180'])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['180'])][info_to_state(x-1, y, ORIENTATIONS['180'])] = I_ACTIONS['FORWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['180'])][info_to_state(x, y-1, ORIENTATIONS['240'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['180'])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = I_ACTIONS['BIG_TURN']

            # 240 Degrees
            Arena[info_to_state(x, y, ORIENTATIONS['240'])][info_to_state(x+1, y, ORIENTATIONS['0'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['240'])][info_to_state(x, y+1, ORIENTATIONS['60'])] = I_ACTIONS['BACKWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['240'])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['240'])][info_to_state(x-1, y, ORIENTATIONS['180'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['240'])][info_to_state(x, y-1, ORIENTATIONS['240'])] = I_ACTIONS['FORWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['240'])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = I_ACTIONS['SMALL_TURN']

            # 300 Degrees
            Arena[info_to_state(x, y, ORIENTATIONS['300'])][info_to_state(x+1, y, ORIENTATIONS['0'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['300'])][info_to_state(x, y+1, ORIENTATIONS['60'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['300'])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['BACKWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['300'])][info_to_state(x-1, y, ORIENTATIONS['180'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['300'])][info_to_state(x, y-1, ORIENTATIONS['240'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['300'])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = I_ACTIONS['FORWARD']

    # Mofifying the top edge of grid
    Y_LEVEL = grid_size - 1
    for x in range(1, grid_size - 1):
        y = Y_LEVEL

        # 0 Degrees
        curr_orientation = '0'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['FORWARD']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['BIG_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = PW_ACTIONS['SMALL_TURN']
        
        # 60 Degrees
        curr_orientation = '60'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = CW_ACTIONS['SMALL_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = 
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = CW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = CW_ACTIONS['REFLECT']

        curr_orientation = '120'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = CW_ACTIONS['BIG_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS[]
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = CW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = CW_ACTIONS['BACKWARD']

        curr_orientation = '180'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['BACKWARD']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['FORWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = PW_ACTIONS['BIG_TURN']

        # TODO: Even though not possible for orientations 240 and 300, analyze for possible drone starting points

    # Mofifying the bottom edge of grid
    Y_LEVEL = 0
    for x in range(1, grid_size - 1):
        y = Y_LEVEL

        curr_orientation = '0'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['FORWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['BACKWARD']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = 
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] =
        
        curr_orientation = '180'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['FORWARD']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] =
        
        curr_orientation = '240'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = CW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = CW_ACTIONS['SMALL_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] =
        
        curr_orientation = '300'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = CW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = CW_ACTIONS['BIG_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] =

    # Modifying the left edge of the grid
    X_LEVEL = 0
    for y in range(1, grid_size - 1):
        x = X_LEVEL
        # TODO: Degree 0 and 300
        curr_orientation = '60'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['FORWARD']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] =PW_ACTIONS[]
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = PW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = PW_ACTIONS['BIG_TURN']

        curr_orientation = '120'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS['SMALL_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = CW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = CW_ACTIONS['BACKWARD']

        curr_orientation = '180'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS['BIG_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = CW_ACTIONS[]
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = CW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = CW_ACTIONS['REFLECT']
        
        curr_orientation = '240'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['BACKWARD']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = PW_ACTIONS['FORWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = PW_ACTIONS['SMALL_TURN']
        
    # Modifying the RIGHT edge of the grid
    X_LEVEL = grid_size - 1
    for y in range(1, grid_size - 1):
        x = X_LEVEL
        # TODO: Degree 180 and 120

        curr_orientation = '0'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = CW_ACTIONS['BIG_TURN']

        curr_orientation = '60'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['FORWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = PW_ACTIONS['BACKWARD']
        
        curr_orientation = '240'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = PW_ACTIONS['FORWARD']
        
        curr_orientation = '300'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = CW_ACTIONS['SMALL_TURN']
    
    # Top left corner
    x = 0
    y = grid_size - 1
    
    curr_orientation = '0'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = 0.25
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = 0.25

    curr_orientation = '60'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = AOC_ACTIONS['SMALL_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = AOC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = AOC_ACTIONS['BIG_TURN']
    
    curr_orientation = '120'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = COC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = COC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = COC_ACTIONS['BACKWARD']
    
    curr_orientation = '180'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = AOC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = AOC_ACTIONS['SMALL_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = AOC_ACTIONS['BIG_TURN']
    
    curr_orientation = '240'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = 0.25
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = 0.25
    
    curr_orientation = '300'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = 0.25
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = 0.25
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = 0.5

    # TOP RIGHT CORNER
    x = grid_size - 1
    y = grid_size - 1
    
    # TODO: ADD PROBABILTIIES FOR OTHER ORIENTATIONS 120, 180, 240, 300
    curr_orientation = '0'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = ASC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = ASC_ACTIONS['BIG_TURN']
    
    curr_orientation = '60'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = ASC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = ASC_ACTIONS['BACKWARD']

    # BOTTOM RIGHT CORNER
    x = grid_size - 1
    y = 0
    curr_orientation = '0'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = AOC_ACTIONS['SMALL_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = AOC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = AOC_ACTIONS['BACKWARD']
    
    curr_orientation = '60'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = 0.25
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = 0.25
    
    curr_orientation = '120'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.25
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = 0.25
    
    curr_orientation = '180'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.25
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = 0.25
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = 0.5
    
    curr_orientation = '240'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = AOC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = AOC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = AOC_ACTIONS['SMALL_TURN']
    
    curr_orientation = '300'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = COC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] = COC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = COC_ACTIONS['BIG_TURN']

    # Bottom left corner
    x = 0
    y = 0
    
    curr_orientation = '0'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.5
    
    curr_orientation = '60'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.5
    
    curr_orientation = '120'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.5
    
    curr_orientation = '180'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = ASC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = ASC_ACTIONS['BIG_TURN']
                                                                                                                        
    curr_orientation = '240'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = ASC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = ASC_ACTIONS['BACKWARD']

    curr_orientation = '300'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.5

    # Modifying for hops to target, this assumes that the target has a one hop border
    x, y = target_pos
    for orientation in range(6):
        Arena[info_to_state(x+1, y, orientation)] = 0
        Arena[info_to_state(x+1, y, orientation)][info_to_state(x, y, orientation)] = 1
        
        Arena[info_to_state(x-1, y, orientation)] = 0
        Arena[info_to_state(x-1, y, orientation)][info_to_state(x, y, orientation)] = 1

        Arena[info_to_state(x-1, y+1, orientation)] = 0
        Arena[info_to_state(x-1, y+1, orientation)][info_to_state(x, y, orientation)] = 1

        Arena[info_to_state(x, y+1, orientation)] = 0
        Arena[info_to_state(x, y+1, orientation)][info_to_state(x, y, orientation)] = 1
        
        Arena[info_to_state(x+1, y-1, orientation)] = 0
        Arena[info_to_state(x+1, y-1, orientation)][info_to_state(x, y, orientation)] = 1

        Arena[info_to_state(x, y-1, orientation)] = 0
        Arena[info_to_state(x, y-1, orientation)][info_to_state(x, y, orientation)] = 1
    return Arena


'''
Although not used in the acutal simulation, it is used to get information for graphing purposes
'''
def state_to_info(state, grid_size):
    orientation = state % 6
    state //= 6
    y = state // grid_size
    x = state % grid_size
    return x, y, orientation

def get_robot_next_move(curr_state, probability_matrix):
    p = probability_matrix[curr_state]
    return np.random.choice(len(p), p=p)

def start_robot_swarming(grid_size, target_pos):
    probability_matrix = init_arena(grid_size, target_pos)

    def info_to_state(x, y, orientation):
        return 6 * (x + grid_size * y) + orientation
    
    curr_state = info_to_state(1, 0, 0)
    history = []
    while True:
        x, y, _ = state_to_info(curr_state, grid_size)
        history.append((x, y))
        if x == target_pos[0] and y == target_pos[1]:
            print("FOUND TARGET AT ({}, {})".format(x, y))
            break
        print("Robot is now at ({}, {})".format(x, y))
        curr_state = get_robot_next_move(curr_state, probability_matrix)
    graph_swarm.graph_arena(grid_size, target_pos, history)

start_robot_swarming(20, (5, 5))