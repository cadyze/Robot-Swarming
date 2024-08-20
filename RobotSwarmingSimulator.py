import numpy as np
import graph_swarm

from enum import Enum

class COLLISION_PROTOCOL(Enum):
    BREAK = 1
    FIND_NEXT_AVAILABLE = 2


class SwarmSimulator:
    def __init__(self, grid_size, target_pos, obstacles):
        self.grid_size = grid_size
        self.target_pos = target_pos
        self.collisions = 0
        self.Arena, self.DynamicObstacleArena = self.init_arena(grid_size, target_pos, obstacles)
        self.obstacles = obstacles

    def init_arena(self, grid_size, target_pos, obstacles): 
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
            curr_orientation = '0'
            Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y, ORIENTATIONS['0'])] = 0.4
            Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y+1, ORIENTATIONS['60'])] = 0.2
            # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y+1, ORIENTATIONS['120'])] =PW_ACTIONS[]
            # Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x-1, y, ORIENTATIONS['180'])] = PW_ACTIONS
            Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x, y-1, ORIENTATIONS['240'])] = 0.2
            Arena[info_to_state(x, y, ORIENTATIONS[curr_orientation])][info_to_state(x+1, y-1, ORIENTATIONS['300'])] = 0.2

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

        DyanmicObstacleArena = Arena.copy()
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
                    DyanmicObstacleArena[info_to_state(x, y, orientation)] = 0
                    for node in available_nodes:
                        Arena[info_to_state(x, y, orientation)][info_to_state(node[0], node[1], orientation)] = wall_prob
                        DyanmicObstacleArena[info_to_state(x, y, orientation)][info_to_state(node[0], node[1], orientation)] = wall_prob

        return Arena, DyanmicObstacleArena


    '''
    Although not used in the acutal simulation, it is used to get information for graphing purposes
    '''
    def state_to_info(self, state, grid_size):
        orientation = state % 6
        state //= 6
        y = state // grid_size
        x = state % grid_size
        return x, y, orientation

    def get_robot_next_move(self, curr_states, robot_ind, probability_matrix, grid_size, current_positions, collision_protocol):
        p = probability_matrix[curr_states[robot_ind]]
        next_state = np.random.choice(len(p), p=p)
        x, y, _ = self.state_to_info(next_state, grid_size)
            
        if (x, y) in current_positions:
            # Run a specified collision protocol
            self.collisions += 1

            if collision_protocol == COLLISION_PROTOCOL.BREAK:
                # If collding with another robot, just return its current state
                next_state = curr_states[robot_ind]
                x, y, _ = self.state_to_info(next_state, grid_size)
            elif collision_protocol == COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE:
                max_iters = 100
                for _ in range(max_iters):
                    next_state = np.random.choice(len(p), p=p)
                    x, y, _ = self.state_to_info(next_state, grid_size)
                    if (x, y) not in current_positions:
                        break
        return next_state, (x, y)

    def start_robot_swarming(self, num_robots, collision_protocol, 
                            show_graph=True, tracked_robot=-1, wait_for_all=False):
        
        if wait_for_all and tracked_robot != -1:
            raise Exception("ERROR: CANNOT TRACK A ROBOT WHILE WAITING FOR ALL TO FINISH.")
            
        collisions = 0

        def info_to_state(x, y, orientation):
            return 6 * (x + self.grid_size * y) + orientation
        
        # Instantiate the lower left corner with number of drones starting at 0, 0
        curr_states = [info_to_state(0, 0, 0)]
        n_to_instantiate = num_robots - 1
        x, y = 0, 1
        layer = 1
        while n_to_instantiate != 0:
            # Instantiate a new robot
            curr_states.append(info_to_state(x, y, 0))
            n_to_instantiate -= 1

            # Update for next coordinates
            if y == 0:
                layer += 1
                y = layer
                x = 0
            else:
                x += 1
                y -= 1

        # Reverse so that the first robot is able to move out freely
        curr_states = curr_states[::-1]

        curr_states = curr_states[:num_robots]
        history = [[] for _ in range(num_robots)]
        is_target_found = False
        current_positions = [(0, 0) for _ in range(num_robots)]
        robots_finished = [False for _ in range(num_robots)]

        num_moves = 0
        while not is_target_found:
            for robot_ind in range(num_robots):
                x, y, _ = self.state_to_info(curr_states[robot_ind], self.grid_size)
                history[robot_ind].append((x, y))

                if wait_for_all:
                    if x == self.target_pos[0] and y == self.target_pos[1]:
                        # All finished robots will be relocated to off the board to avoid extra collisions
                        robots_finished[robot_ind] = True
                        current_positions[robot_ind] = (-1, -1)
                    else:
                        curr_states[robot_ind], current_positions[robot_ind] = self.get_robot_next_move(curr_states, robot_ind, self.Arena, self.grid_size, current_positions, collision_protocol)
                else:
                    # Code for tracking robots
                    if tracked_robot != -1:
                        if tracked_robot == robot_ind:
                            if x == self.target_pos[0] and y == self.target_pos[1]:
                                is_target_found = True
                            curr_states[robot_ind], current_positions[robot_ind] = self.get_robot_next_move(curr_states, robot_ind, self.Arena, self.grid_size, current_positions, collision_protocol)
                        else:
                            curr_states[robot_ind], current_positions[robot_ind] = self.get_robot_next_move(curr_states, robot_ind, self.DynamicObstacleArena, self.grid_size, current_positions, collision_protocol)
                    else:
                        curr_states[robot_ind], current_positions[robot_ind] = self.get_robot_next_move(curr_states, robot_ind, self.Arena, self.grid_size, current_positions, collision_protocol)
                        if x == self.target_pos[0] and y == self.target_pos[1]:
                            is_target_found = True
            
            # Check goals for wait for all
            if wait_for_all and all(robots_finished):
                is_target_found = True
            num_moves += 1

        if show_graph:
            graph_swarm.graph_arena(self.grid_size, self.target_pos, history, self.obstacles, tracked_robot)
        return num_moves, self.collisions