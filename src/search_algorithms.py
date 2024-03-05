from collections import deque
from globals import *
from heuristics import euclidean_distance

def bfs_path(maze, start: tuple, end: tuple):
    queue = deque([(start, [start])])
    visited = set()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    iterations = 0
    while queue:
        current_node, path = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)
            iterations += 1
            
        if current_node == end:
            return path, iterations
        
        neighbors = []
        for direction in directions:
            temp_i = current_node[0] + direction[0]
            temp_j = current_node[1] + direction[1]
            if (0 <= temp_i < len(maze)) and (0 <= temp_j < len(maze[temp_i])) and maze[temp_i][temp_j] != WALL and (temp_i, temp_j) not in visited:
                neighbors.append((temp_i, temp_j))
        
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    return None, iterations

def dfs_path(maze, start: tuple, end: tuple):
    stack = [(start, [start])]
    visited = set()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    iterations = 0
    while stack:
        current_node, path = stack.pop()
        if current_node not in visited:
            visited.add(current_node)
            iterations += 1
            
        if current_node == end:
            return path, iterations
        
        neighbors = []
        for direction in directions:
            temp_i = current_node[0] + direction[0]
            temp_j = current_node[1] + direction[1]
            if temp_i < 0 or temp_i > len(maze) - 1 or temp_j < 0 or temp_j > len(maze[temp_i]) - 1:
                continue
            if maze[temp_i][temp_j] != WALL and (temp_i, temp_j) not in visited:
                neighbors.append((temp_i, temp_j))
        
        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None, iterations

def dls_path(maze, start: tuple, end: tuple, depth_limit: int = 100):
    stack = [(start, [start], 0)]
    visited = set()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    iterations = 0
    while stack:
        current_node, path, depth = stack.pop()
        if current_node not in visited:
            visited.add(current_node)
            iterations += 1
            
        if current_node == end:
            return path, iterations
        
        if depth < depth_limit:
            neighbors = []
            for direction in directions:
                temp_i = current_node[0] + direction[0]
                temp_j = current_node[1] + direction[1]
                if (0 <= temp_i < len(maze)) and (0 <= temp_j < len(maze[temp_i])) and maze[temp_i][temp_j] != WALL and (temp_i, temp_j) not in visited:
                    neighbors.append((temp_i, temp_j))
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], depth + 1))
        else:
            print("Depth limit reached.")
            return None, iterations
    return None, iterations

def a_star_path(maze, start: tuple, end: tuple, heuristic_func=None):
    if heuristic_func is None:
        heuristic_func = euclidean_distance  # Default heuristic function

    def heuristic_cost(node):
        return heuristic_func(node, end)

    queue = deque([(start, [start], 0)])
    visited = set()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    iterations = 0
    while queue:
        current_node, path, cost = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)
            iterations += 1
            
        if current_node == end:
            return path, iterations
        
        neighbors = []
        for direction in directions:
            temp_i = current_node[0] + direction[0]
            temp_j = current_node[1] + direction[1]
            if temp_i < 0 or temp_i > len(maze) - 1 or temp_j < 0 or temp_j > len(maze[temp_i]) - 1:
                continue
            if maze[temp_i][temp_j] != WALL and (temp_i, temp_j) not in visited:
                neighbors.append((temp_i, temp_j))
        
        for neighbor in neighbors:
            if neighbor not in visited:
                new_cost = cost + 1  # Assuming each step has a cost of 1
                heuristic = heuristic_cost(neighbor)  # Calculate heuristic cost for the neighbor
                queue.append((neighbor, path + [neighbor], new_cost + heuristic))  # Add heuristic to total cost
                # Sort the queue based on the total cost
                queue = deque(sorted(queue, key=lambda x: x[2]))
    return None, iterations

def greedy_best_first_search(maze, start: tuple, end: tuple, heuristic_func=None):
    if heuristic_func is None:
        heuristic_func = euclidean_distance  # Default heuristic function

    def heuristic_cost(node):
        return heuristic_func(node, end)

    queue = deque([(start, [start], heuristic_cost(start))])
    visited = set()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    iterations = 0
    while queue:
        current_node, path, current_heuristic = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)
            iterations += 1
            
        if current_node == end:
            return path, iterations
        
        neighbors = []
        for direction in directions:
            temp_i = current_node[0] + direction[0]
            temp_j = current_node[1] + direction[1]
            if (0 <= temp_i < len(maze)) and (0 <= temp_j < len(maze[temp_i])) and maze[temp_i][temp_j] != WALL and (temp_i, temp_j) not in visited:
                neighbors.append((temp_i, temp_j))
        
        for neighbor in neighbors:
            if neighbor not in visited:
                heuristic = heuristic_cost(neighbor)  # Calculate heuristic cost for the neighbor
                queue.append((neighbor, path + [neighbor], heuristic))  # Add heuristic cost to the queue
                # Sort the queue based on the heuristic cost
                queue = deque(sorted(queue, key=lambda x: x[2]))
    return None, iterations
