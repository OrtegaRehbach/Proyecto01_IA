import math

def euclidean_distance(pos1, pos2):
    """
    Calculate the Euclidean distance between two positions in the maze matrix.

    Arguments:
    pos1 (tuple): (x, y) coordinates of the first position.
    pos2 (tuple): (x, y) coordinates of the second position.

    Returns:
    float: Euclidean distance between the two positions.
    """
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def manhattan_distance(pos1, pos2):
    """
    Calculate the Manhattan distance between two positions in the maze matrix.

    Arguments:
    pos1 (tuple): (x, y) coordinates of the first position.
    pos2 (tuple): (x, y) coordinates of the second position.

    Returns:
    int: Manhattan distance between the two positions.
    """
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x2 - x1) + abs(y2 - y1)
