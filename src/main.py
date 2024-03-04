import matplotlib.pyplot as plt
import numpy as np
from collections import deque

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

def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print( str(matrix[i][j]).ljust(2),end=' ')
        print()

path = dls_path(start, end)
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
