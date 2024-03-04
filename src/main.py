import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from maze_file_reader import MazeFileReader
from search_algorithms import bfs_path, dfs_path, dls_path, a_star_path, greedy_best_first_search
from heuristics import euclidean_distance, manhattan_distance

WALL = 1
PATH = 0
START = 2
END = 3

def print_menu():
    print("Choose a search algorithm:")
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search (DFS)")
    print("3. Depth-Limited Search (DLS)")
    print("4. A* Search")
    print("5. Greedy Best-First Search")
    print("6. Exit")

def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(str(matrix[i][j]).ljust(2), end=' ')
        print()

def run_search_algorithm(algorithm, maze, start, end):
    if algorithm == 1:
        return bfs_path(maze, start, end)
    elif algorithm == 2:
        return dfs_path(maze, start, end)
    elif algorithm == 3:
        depth_limit = int(input("Enter depth limit for DLS: "))
        return dls_path(maze, start, end, depth_limit)
    elif algorithm == 4:
        return a_star_path(maze, start, end, heuristic_func=euclidean_distance)
    elif algorithm == 5:
        return greedy_best_first_search(maze, start, end, heuristic_func=euclidean_distance)
    elif algorithm == 6:
        print("Exiting...")
        return None
    else:
        print("Invalid choice")
        return None

def main():
    maze_file_name = "simple_test_maze.txt"
    reader = MazeFileReader(f"./res/{maze_file_name}")
    test_maze = reader.get_maze()

    start = None
    end = None
    for i in range(len(test_maze)):
        for j in range(len(test_maze[i])):
            if test_maze[i][j] == START:
                start = (i, j)
            if test_maze[i][j] == END:
                end = (i, j)
    if not start:
        print(f"Maze in '{maze_file_name}' does not specify a starting position.")
    if not end:
        print(f"Maze in '{maze_file_name}' does not specify an ending position.")
    if not start or not end:
        return
    test_maze[start[0]][start[1]] = 0
    test_maze[end[0]][end[1]] = 0

    while True:
        print_menu()
        choice = int(input("Enter your choice: "))

        if choice == 6:
            break

        path = run_search_algorithm(choice, test_maze, start, end)

        if path:
            print("Path:", path)
            k = 1
            for step_i, step_j in path:
                if test_maze[step_i][step_j] == PATH:
                    test_maze[step_i][step_j] = k
                    k += 1
            print_matrix(test_maze)
            np_array = np.array(test_maze)
            plt.matshow(np_array)
            plt.show()

if __name__ == "__main__":
    main()
