"""
Builds the swarm's hex-orientation transition matrix (Arena) and its
dynamic-obstacle variant, as sparse CSR matrices.

Mechanically ported from RobotSwarmingSimulator.SwarmSimulator.init_arena:
same per-cell transition probabilities (interior / edge / corner cases,
target absorption, sensing-range hops, obstacle walls), just backed by
scipy.sparse instead of a dense ndarray so large grids (e.g. 61x61, which
would be a ~4GB dense matrix) stay cheap to build, copy, and sample from.
"""
import numpy as np
from scipy.sparse import lil_matrix


def build_arena(grid_size, target_pos, obstacles, laziness_prob, sensing_range, for_math=False):
    nl_sc = 1 - laziness_prob
    # Six different matrices, each representing the probability matrix for each orientation (measured in degrees)
    I_ACTIONS = {'FORWARD': 0.49 * nl_sc, 'BACKWARD': 0.01 * nl_sc, 'SMALL_TURN': 0.24 * nl_sc, 'BIG_TURN': 0.01 * nl_sc}	# I: Inside
    CW_ACTIONS = {'REFLECT': 0.6 * nl_sc, 'BACKWARD': 0.15 * nl_sc, 'SMALL_TURN': 0.15 * nl_sc, 'BIG_TURN': 0.1 * nl_sc}	# CW: Collide against the wall
    PW_ACTIONS = {'FORWARD': 0.45 * nl_sc, 'BACKWARD': 0.04 * nl_sc, 'SMALL_TURN': 0.45 * nl_sc, 'BIG_TURN': 0.06 * nl_sc}	# PW: Parallel with the wall
    ASC_ACTIONS = {'BACKWARD': 0.5 * nl_sc, 'BIG_TURN': 0.5 * nl_sc}										                # ASC: Along the sharp corner
    AOC_ACTIONS = {'BACKWARD': 0.25 * nl_sc, 'SMALL_TURN': 0.25 * nl_sc, 'BIG_TURN': 0.5 * nl_sc}					        # AOC: Along the obtuse corner
    COC_ACTIONS = {'BACKWARD': 0.34 * nl_sc, 'BIG_TURN': 0.33 * nl_sc}										                # COC: Collide toward the obtuse corner

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
    if for_math:
        arena_size += 1

    Arena = lil_matrix((arena_size, arena_size))


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
            Arena[info_to_state(x, y, ORIENTATIONS['0']), info_to_state(x+1, y, ORIENTATIONS['0'])] = I_ACTIONS['FORWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['0']), info_to_state(x, y+1, ORIENTATIONS['60'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['0']), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['0']), info_to_state(x-1, y, ORIENTATIONS['180'])] = I_ACTIONS['BACKWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['0']), info_to_state(x, y-1, ORIENTATIONS['240'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['0']), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['0']), info_to_state(x, y, ORIENTATIONS['0'])] = laziness_prob


            # 60 Degrees
            Arena[info_to_state(x, y, ORIENTATIONS['60']), info_to_state(x+1, y, ORIENTATIONS['0'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['60']), info_to_state(x, y+1, ORIENTATIONS['60'])] = I_ACTIONS['FORWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['60']), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['60']), info_to_state(x-1, y, ORIENTATIONS['180'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['60']), info_to_state(x, y-1, ORIENTATIONS['240'])] = I_ACTIONS['BACKWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['60']), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['60']), info_to_state(x, y, ORIENTATIONS['60'])] = laziness_prob

            # 120 Degrees
            Arena[info_to_state(x, y, ORIENTATIONS['120']), info_to_state(x+1, y, ORIENTATIONS['0'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['120']), info_to_state(x, y+1, ORIENTATIONS['60'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['120']), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['FORWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['120']), info_to_state(x-1, y, ORIENTATIONS['180'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['120']), info_to_state(x, y-1, ORIENTATIONS['240'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['120']), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = I_ACTIONS['BACKWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['120']), info_to_state(x, y, ORIENTATIONS['120'])] = laziness_prob

            # 180 Degrees
            Arena[info_to_state(x, y, ORIENTATIONS['180']), info_to_state(x+1, y, ORIENTATIONS['0'])] = I_ACTIONS['BACKWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['180']), info_to_state(x, y+1, ORIENTATIONS['60'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['180']), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['180']), info_to_state(x-1, y, ORIENTATIONS['180'])] = I_ACTIONS['FORWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['180']), info_to_state(x, y-1, ORIENTATIONS['240'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['180']), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['180']), info_to_state(x, y, ORIENTATIONS['180'])] = laziness_prob

            # 240 Degrees
            Arena[info_to_state(x, y, ORIENTATIONS['240']), info_to_state(x+1, y, ORIENTATIONS['0'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['240']), info_to_state(x, y+1, ORIENTATIONS['60'])] = I_ACTIONS['BACKWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['240']), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['240']), info_to_state(x-1, y, ORIENTATIONS['180'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['240']), info_to_state(x, y-1, ORIENTATIONS['240'])] = I_ACTIONS['FORWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['240']), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['240']), info_to_state(x, y, ORIENTATIONS['240'])] = laziness_prob

            # 300 Degrees
            Arena[info_to_state(x, y, ORIENTATIONS['300']), info_to_state(x+1, y, ORIENTATIONS['0'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['300']), info_to_state(x, y+1, ORIENTATIONS['60'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['300']), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['BACKWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['300']), info_to_state(x-1, y, ORIENTATIONS['180'])] = I_ACTIONS['BIG_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['300']), info_to_state(x, y-1, ORIENTATIONS['240'])] = I_ACTIONS['SMALL_TURN']
            Arena[info_to_state(x, y, ORIENTATIONS['300']), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = I_ACTIONS['FORWARD']
            Arena[info_to_state(x, y, ORIENTATIONS['300']), info_to_state(x, y, ORIENTATIONS['300'])] = laziness_prob

    # Mofifying the top edge of grid
    Y_LEVEL = grid_size - 1
    for x in range(1, grid_size - 1):
        y = Y_LEVEL

        # 0 Degrees
        curr_orientation = '0'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['FORWARD']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['BIG_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = I_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        # 60 Degrees
        curr_orientation = '60'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = CW_ACTIONS['SMALL_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = 
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = CW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '120'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = CW_ACTIONS['BIG_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS[]
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = CW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '180'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['BACKWARD']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['FORWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        # TODO: Even though not possible for orientations 240 and 300, analyze for possible drone starting points
        curr_orientation = '240'
        move_prob = (1 - laziness_prob) / 8
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = move_prob
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = move_prob * 2
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = move_prob * 3
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = move_prob * 2
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '300'
        move_prob = (1 - laziness_prob) / 8
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = move_prob * 2
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = move_prob
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = move_prob * 2
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = move_prob * 3
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    # Mofifying the bottom edge of grid
    Y_LEVEL = 0
    for x in range(1, grid_size - 1):
        y = Y_LEVEL

        curr_orientation = '60'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['FORWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '120'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['FORWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob



        curr_orientation = '0'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['FORWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '180'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['FORWARD']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '240'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = CW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = CW_ACTIONS['SMALL_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '300'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = CW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = CW_ACTIONS['BIG_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    # Modifying the left edge of the grid
    X_LEVEL = 0
    for y in range(1, grid_size - 1):
        x = X_LEVEL
        # TODO: Degree 0 and 300
        curr_orientation = '0'
        move_prob = (1 - laziness_prob) / 8
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = move_prob * 3
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = move_prob * 2
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = move_prob
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = move_prob * 2
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '300'
        move_prob = (1 - laziness_prob) / 8
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = move_prob * 2
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = move_prob
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = move_prob * 2
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = move_prob * 3
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '60'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['FORWARD']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] =PW_ACTIONS[]
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = PW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '120'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS['SMALL_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = CW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '180'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS['BIG_TURN']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = CW_ACTIONS[]
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = CW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '240'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['BACKWARD']
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] =
        # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] =
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = PW_ACTIONS['FORWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    # Modifying the RIGHT edge of the grid
    X_LEVEL = grid_size - 1
    for y in range(1, grid_size - 1):
        x = X_LEVEL
        # TODO: Degree 180 and 120
        move_prob = (1 - laziness_prob) / 8
        curr_orientation = '180'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = move_prob
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = move_prob * 2
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = move_prob * 3
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = move_prob * 2
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '120'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = move_prob * 2
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = move_prob * 3
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = move_prob * 2
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = move_prob
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '0'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = CW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '60'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['FORWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = PW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '240'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = PW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = PW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = PW_ACTIONS['FORWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

        curr_orientation = '300'
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = CW_ACTIONS['BIG_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = CW_ACTIONS['BACKWARD']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = CW_ACTIONS['REFLECT']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = CW_ACTIONS['SMALL_TURN']
        Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    # Top left corner
    x = 0
    y = grid_size - 1

    curr_orientation = '0'
    move_prob = (1 - laziness_prob) / 4
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = move_prob * 2
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = move_prob
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = move_prob
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    curr_orientation = '60'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = AOC_ACTIONS['SMALL_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = AOC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = AOC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    curr_orientation = '120'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = COC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = COC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = COC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob    

    curr_orientation = '180'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = AOC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = AOC_ACTIONS['SMALL_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = AOC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    curr_orientation = '240'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = move_prob
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = move_prob * 2
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = move_prob
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    curr_orientation = '300'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = move_prob
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = move_prob
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y-1, ORIENTATIONS['300'])] = move_prob * 2
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    # TOP RIGHT CORNER
    x = grid_size - 1
    y = grid_size - 1

    # TODO: ADD PROBABILTIIES FOR OTHER ORIENTATIONS 120, 180, 240, 300
    curr_orientation = '120'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = ASC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = ASC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    curr_orientation = '180'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = ASC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = ASC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    curr_orientation = '240'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = ASC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = ASC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    curr_orientation = '300'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = ASC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = ASC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    curr_orientation = '0'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = ASC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = ASC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    curr_orientation = '60'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = ASC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y-1, ORIENTATIONS['240'])] = ASC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    # BOTTOM RIGHT CORNER
    x = grid_size - 1
    y = 0
    curr_orientation = '0'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = AOC_ACTIONS['SMALL_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = AOC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = AOC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    curr_orientation = '60'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = 0.25
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = 0.25

    curr_orientation = '120'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.25
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = 0.25

    curr_orientation = '180'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.25
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = 0.25
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = 0.5

    curr_orientation = '240'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = AOC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = AOC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = AOC_ACTIONS['SMALL_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    curr_orientation = '300'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = COC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y+1, ORIENTATIONS['120'])] = COC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x-1, y, ORIENTATIONS['180'])] = COC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    # Bottom left corner
    x = 0
    y = 0

    curr_orientation = '0'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.5

    curr_orientation = '60'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.5

    curr_orientation = '120'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.5

    curr_orientation = '180'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = ASC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = ASC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    curr_orientation = '240'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = ASC_ACTIONS['BIG_TURN']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = ASC_ACTIONS['BACKWARD']
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    curr_orientation = '300'
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x+1, y, ORIENTATIONS['0'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.5
    Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation]), info_to_state(x, y, ORIENTATIONS[curr_orientation])] = laziness_prob

    DynamicObstacleArena = Arena.copy()
    # Modifying for hops to target, this assumes that the target has a one hop border
    x, y = target_pos
    # coords_to_modify = [(x, y)]

    for orientation in range(6):
        Arena[info_to_state(x+1, y, orientation)] = 0
        Arena[info_to_state(x+1, y, orientation), info_to_state(x, y, orientation)] = 1

        Arena[info_to_state(x-1, y, orientation)] = 0
        Arena[info_to_state(x-1, y, orientation), info_to_state(x, y, orientation)] = 1

        Arena[info_to_state(x-1, y+1, orientation)] = 0
        Arena[info_to_state(x-1, y+1, orientation), info_to_state(x, y, orientation)] = 1

        Arena[info_to_state(x, y+1, orientation)] = 0
        Arena[info_to_state(x, y+1, orientation), info_to_state(x, y, orientation)] = 1

        Arena[info_to_state(x+1, y-1, orientation)] = 0
        Arena[info_to_state(x+1, y-1, orientation), info_to_state(x, y, orientation)] = 1

        Arena[info_to_state(x, y-1, orientation)] = 0
        Arena[info_to_state(x, y-1, orientation), info_to_state(x, y, orientation)] = 1

        if sensing_range >= 2:
            Arena[info_to_state(x-2, y, orientation)] = 0
            Arena[info_to_state(x-2, y, orientation), info_to_state(x-1, y, orientation)] = 1

            Arena[info_to_state(x-1, y-1, orientation)] = 0
            Arena[info_to_state(x-1, y-1, orientation), info_to_state(x-1, y, orientation)] = 1

            Arena[info_to_state(x-2, y+1, orientation)] = 0
            Arena[info_to_state(x-2, y+1, orientation), info_to_state(x-1, y, orientation)] = 1

            Arena[info_to_state(x-2, y+2, orientation)] = 0
            Arena[info_to_state(x-2, y+2, orientation), info_to_state(x-1, y+1, orientation)] = 1

            Arena[info_to_state(x-1, y+2, orientation)] = 0
            Arena[info_to_state(x-1, y+2, orientation), info_to_state(x-1, y+1, orientation)] = 1

            Arena[info_to_state(x, y+2, orientation)] = 0
            Arena[info_to_state(x, y+2, orientation), info_to_state(x, y+1, orientation)] = 1

            Arena[info_to_state(x+1, y+1, orientation)] = 0
            Arena[info_to_state(x+1, y+1, orientation), info_to_state(x, y+1, orientation)] = 1

            #TODO: Add connections
            Arena[info_to_state(x+2, y, orientation)] = 0
            Arena[info_to_state(x+2, y, orientation), info_to_state(x+1, y, orientation)] = 1

            Arena[info_to_state(x+2, y-1, orientation)] = 0
            Arena[info_to_state(x+2, y-1, orientation), info_to_state(x, y+1, orientation)] = 1

            Arena[info_to_state(x+2, y-2, orientation)] = 0
            Arena[info_to_state(x+2, y-2, orientation), info_to_state(x, y+1, orientation)] = 1

            Arena[info_to_state(x+1, y-2, orientation)] = 0
            Arena[info_to_state(x+1, y-2, orientation), info_to_state(x, y+1, orientation)] = 1

            Arena[info_to_state(x, y-2, orientation)] = 0
            Arena[info_to_state(x, y-2, orientation), info_to_state(x, y+1, orientation)] = 1

        if sensing_range >= 3:
            Arena[info_to_state(x, y+3, orientation)] = 0
            Arena[info_to_state(x, y+3, orientation), info_to_state(x, y+2, orientation)] = 1

            Arena[info_to_state(x+1, y+2, orientation)] = 0
            Arena[info_to_state(x+1, y+2, orientation), info_to_state(x-1, y+2, orientation)] = 1

            Arena[info_to_state(x+2, y+1, orientation)] = 0
            Arena[info_to_state(x+2, y+1, orientation), info_to_state(x+1, y+1, orientation)] = 1

            Arena[info_to_state(x+3, y, orientation)] = 0
            Arena[info_to_state(x+3, y, orientation), info_to_state(x+2, y, orientation)] = 1

            Arena[info_to_state(x+3, y-1, orientation)] = 0
            Arena[info_to_state(x+3, y-1, orientation), info_to_state(x+2, y-1, orientation)] = 1

            Arena[info_to_state(x+3, y-2, orientation)] = 0
            Arena[info_to_state(x+3, y-2, orientation), info_to_state(x+2, y-2, orientation)] = 1

            Arena[info_to_state(x+3, y-3, orientation)] = 0
            Arena[info_to_state(x+3, y-3, orientation), info_to_state(x+2, y-2, orientation)] = 1

            Arena[info_to_state(x+2, y-3, orientation)] = 0
            Arena[info_to_state(x+2, y-3, orientation), info_to_state(x+2, y-2, orientation)] = 1

            Arena[info_to_state(x+1, y-3, orientation)] = 0
            Arena[info_to_state(x+1, y-3, orientation), info_to_state(x+1, y-2, orientation)] = 1

            Arena[info_to_state(x, y-3, orientation)] = 0
            Arena[info_to_state(x, y-3, orientation), info_to_state(x+1, y-2, orientation)] = 1

            Arena[info_to_state(x-1, y-2, orientation)] = 0
            Arena[info_to_state(x-1, y-2, orientation), info_to_state(x, y-2, orientation)] = 1

            Arena[info_to_state(x-2, y-1, orientation)] = 0
            Arena[info_to_state(x-2, y-1, orientation), info_to_state(x-1, y-1, orientation)] = 1

            Arena[info_to_state(x-3, y, orientation)] = 0
            Arena[info_to_state(x-3, y, orientation), info_to_state(x-2, y, orientation)] = 1

            Arena[info_to_state(x-3, y+1, orientation)] = 0
            Arena[info_to_state(x-3, y+1, orientation), info_to_state(x-2, y+1, orientation)] = 1

            Arena[info_to_state(x-3, y+2, orientation)] = 0
            Arena[info_to_state(x-3, y+2, orientation), info_to_state(x-2, y+2, orientation)] = 1

            Arena[info_to_state(x-3, y+3, orientation)] = 0
            Arena[info_to_state(x-3, y+3, orientation), info_to_state(x-2, y+2, orientation)] = 1

            Arena[info_to_state(x-2, y+3, orientation)] = 0
            Arena[info_to_state(x-2, y+3, orientation), info_to_state(x-2, y+2, orientation)] = 1

            Arena[info_to_state(x-1, y+3, orientation)] = 0
            Arena[info_to_state(x-1, y+3, orientation), info_to_state(x-1, y+2, orientation)] = 1

        # For the bucket in the sky implementation, add another jump
        if for_math:
            Arena[info_to_state(x, y, orientation)] = 0
            Arena[info_to_state(x, y, orientation), arena_size - 1] = 1
    '''
    Adding obstacles, will modify how the arena works
    Obstacles is a variable that is of [[(int, int)]] containing tuples that represent the nodes that have the nodes on them.
    '''
    for obstacle_nodes in obstacles:

        # Find all the available nodes
        for i in range(len(obstacle_nodes)):
            available_nodes = set()
            # Analyze the node connected behind and in front
            # Find all the available nodes and divide 1 by the number of available nodes to randomize its next move
            curr_wall = obstacle_nodes[i]
            x, y = curr_wall
            prev_wall, next_wall = (0, 0), (0, 0)
            if i == len(obstacle_nodes) - 1:
                prev_wall, next_wall = obstacle_nodes[i-1], obstacle_nodes[0]
            else:
                prev_wall, next_wall = obstacle_nodes[i-1], obstacle_nodes[i+1]
            nodes_to_check = [(x-1, y), (x-1, y+1), (x, y+1), (x+1, y), (x+1, y-1), (x, y-1), (x-1, y)]


            # Find the next wall node going clockwise
            for node in nodes_to_check:
                if node == next_wall:
                    break
                available_nodes.add(node)

            # Find the previous wall node going counter clockwise
            for node in nodes_to_check[::-1]:
                if node == prev_wall:
                    break
                available_nodes.add(node)

            wall_prob = 1.0 / len(available_nodes)
            for orientation in range(6):
                Arena[info_to_state(x, y, orientation)] = 0
                DynamicObstacleArena[info_to_state(x, y, orientation)] = 0
                for node in available_nodes:
                    Arena[info_to_state(x, y, orientation), info_to_state(node[0], node[1], orientation)] = wall_prob
                    DynamicObstacleArena[info_to_state(x, y, orientation), info_to_state(node[0], node[1], orientation)] = wall_prob

    return Arena.tocsr(), DynamicObstacleArena.tocsr()
