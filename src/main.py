import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from maze_file_reader import MazeFileReader
from heuristics import euclidean_distance, manhattan_distance

WALL = 1
PATH = 0
START = 2
END = 3

test_maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 1],
    [1, 0, 0, 1, 0, 1, 1, 0, 3, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

reader = MazeFileReader("./res/test_maze.txt")
test_maze = reader.get_maze()

start = None
end = None
for i in range(len(test_maze)):
    for j in range(len(test_maze[i])):
        if test_maze[i][j] == START:
            start = (i, j)
        if test_maze[i][j] == END:
            end = (i, j)
test_maze[start[0]][start[1]] = 0
test_maze[end[0]][end[1]] = 0

def bfs_path(start: tuple, end: tuple):
    queue = [(start, [start])]
    visited = set()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        current_node, path = queue.pop(0)
        if current_node not in visited:
            visited.add(current_node)
            
        if current_node == end:
            return path
        
        neighbors = []
        for direction in directions:
            temp_i = current_node[0] + direction[0]
            temp_j = current_node[1] + direction[1]
            if temp_i < 0 or temp_i > len(test_maze) - 1 or temp_j < 0 or temp_j > len(test_maze[temp_i]) - 1:
                continue
            if test_maze[temp_i][temp_j] != WALL and (temp_i, temp_j) not in visited:
                neighbors.append((temp_i, temp_j))
        
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    return None

def dfs_path(start: tuple, end: tuple):
    stack = [(start, [start])]
    visited = set()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while stack:
        current_node, path = stack.pop()
        if current_node not in visited:
            visited.add(current_node)
            
        if current_node == end:
            return path
        
        neighbors = []
        for direction in directions:
            temp_i = current_node[0] + direction[0]
            temp_j = current_node[1] + direction[1]
            if temp_i < 0 or temp_i > len(test_maze) - 1 or temp_j < 0 or temp_j > len(test_maze[temp_i]) - 1:
                continue
            if test_maze[temp_i][temp_j] != WALL and (temp_i, temp_j) not in visited:
                neighbors.append((temp_i, temp_j))
        
        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None

def dls_path(start: tuple, end: tuple, depth_limit: int = 100):
    stack = [(start, [start], 0)]
    visited = set()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while stack:
        current_node, path, depth = stack.pop()
        if current_node not in visited:
            visited.add(current_node)
            
        if current_node == end:
            return path
        
        if depth < depth_limit:
            neighbors = []
            for direction in directions:
                temp_i = current_node[0] + direction[0]
                temp_j = current_node[1] + direction[1]
                if temp_i < 0 or temp_i > len(test_maze) - 1 or temp_j < 0 or temp_j > len(test_maze[temp_i]) - 1:
                    continue
                if test_maze[temp_i][temp_j] != WALL and (temp_i, temp_j) not in visited:
                    neighbors.append((temp_i, temp_j))
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], depth + 1))
        else:
            print("Depth limit reached.")
            return None
    return None

def a_star_path(start: tuple, end: tuple, heuristic_func=None):
    if heuristic_func is None:
        heuristic_func = euclidean_distance  # Default heuristic function

    def heuristic_cost(node):
        return heuristic_func(node, end)

    queue = deque([(start, [start], 0)])
    visited = set()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        current_node, path, cost = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)
            
        if current_node == end:
            return path
        
        neighbors = []
        for direction in directions:
            temp_i = current_node[0] + direction[0]
            temp_j = current_node[1] + direction[1]
            if temp_i < 0 or temp_i > len(test_maze) - 1 or temp_j < 0 or temp_j > len(test_maze[temp_i]) - 1:
                continue
            if test_maze[temp_i][temp_j] != WALL and (temp_i, temp_j) not in visited:
                neighbors.append((temp_i, temp_j))
        
        for neighbor in neighbors:
            if neighbor not in visited:
                new_cost = cost + 1  # Assuming each step has a cost of 1
                heuristic = heuristic_cost(neighbor)  # Calculate heuristic cost for the neighbor
                queue.append((neighbor, path + [neighbor], new_cost + heuristic))  # Add heuristic to total cost
                # Sort the queue based on the total cost
                queue = deque(sorted(queue, key=lambda x: x[2]))
    return None

def greedy_best_first_search(start: tuple, end: tuple, heuristic_func=None):
    if heuristic_func is None:
        heuristic_func = euclidean_distance  # Default heuristic function

    def heuristic_cost(node):
        return heuristic_func(node, end)

    queue = deque([(start, [start], heuristic_cost(start))])
    visited = set()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        current_node, path, current_heuristic = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)
            
        if current_node == end:
            return path
        
        neighbors = []
        for direction in directions:
            temp_i = current_node[0] + direction[0]
            temp_j = current_node[1] + direction[1]
            if temp_i < 0 or temp_i > len(test_maze) - 1 or temp_j < 0 or temp_j > len(test_maze[temp_i]) - 1:
                continue
            if test_maze[temp_i][temp_j] != WALL and (temp_i, temp_j) not in visited:
                neighbors.append((temp_i, temp_j))
        
        for neighbor in neighbors:
            if neighbor not in visited:
                heuristic = heuristic_cost(neighbor)  # Calculate heuristic cost for the neighbor
                queue.append((neighbor, path + [neighbor], heuristic))  # Add heuristic cost to the queue
                # Sort the queue based on the heuristic cost
                queue = deque(sorted(queue, key=lambda x: x[2]))
    return None

def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print( str(matrix[i][j]).ljust(2),end=' ')
        print()

path = dls_path(start, end)
# path = a_star_path(start, end, heuristic_func=euclidean_distance)
# path = greedy_best_first_search(start, end, heuristic_func=euclidean_distance)
print("Path:", path)

if path:
    k = 1
    for step_i, step_j in path:
        if test_maze[step_i][step_j] == PATH:
            test_maze[step_i][step_j] = k
            k += 1

print_matrix(test_maze)
    
np_array = np.array(test_maze)
plt.matshow(np_array)
plt.show()
