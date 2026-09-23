from enum import Enum


class COLLISION_PROTOCOL(Enum):
    WAIT_NEXT = 1
    FIND_NEXT_AVAILABLE = 2


class STARTING_POSITION(Enum):
    FILL = 1
    SPACED = 2
    EDGE = 3
