import time
import matplotlib.pyplot as plt
import numpy as np

from globals import *
from maze_file_reader import MazeFileReader
from search_algorithms import bfs_path, dfs_path, dls_path, a_star_path, greedy_best_first_search
from heuristics import euclidean_distance, manhattan_distance

MENU_DIVIDER = "-" * 64

def print_menu(maze_file_name, chosen_heuristic):
    print(MENU_DIVIDER)
    print("Reading maze from:", maze_file_name)
    print("Current heuristic function:", chosen_heuristic.__name__ if callable(chosen_heuristic) else "None")
    print(MENU_DIVIDER)
    print("Choose a search algorithm:")
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search (DFS)")
    print("3. Depth-Limited Search (DLS)")
    print("4. A* Search")
    print("5. Greedy Best-First Search")
    print("6. Choose heuristic function for informed search algorithms")
    print("7. Exit")
    print(MENU_DIVIDER)

def print_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(str(matrix[i][j]).ljust(2), end=' ')
        print()

def run_search_algorithm(algorithm, maze, start, end, chosen_heuristic):
    if algorithm == 1:
        return bfs_path(maze, start, end)
    elif algorithm == 2:
        return dfs_path(maze, start, end)
    elif algorithm == 3:
        depth_limit = int(input("Enter depth limit for DLS: "))
        print(MENU_DIVIDER)
        return dls_path(maze, start, end, depth_limit)
    elif algorithm == 4:
        return a_star_path(maze, start, end, heuristic_func=chosen_heuristic)
    elif algorithm == 5:
        return greedy_best_first_search(maze, start, end, heuristic_func=chosen_heuristic)
    elif algorithm == 6:
        print("Choose a heuristic function:")
        print(MENU_DIVIDER)
        print("1. Euclidean Distance")
        print("2. Manhattan Distance")
        choice = int(input("Enter your choice: "))
        print(MENU_DIVIDER)
        if choice == 1:
            return euclidean_distance
        elif choice == 2:
            return manhattan_distance
        else:
            print("Invalid choice")
            return None
    elif algorithm == 7:
        print("Exiting...")
        return None
    else:
        print("Invalid choice")
        return None

def main():
    maze_file_name = "test_maze.txt"
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
    test_maze[start[0]][start[1]] = PATH
    test_maze[end[0]][end[1]] = PATH

    chosen_heuristic = euclidean_distance  # Default heuristic function

    while True:
        print_menu(maze_file_name, chosen_heuristic)
        choice = int(input("Enter your choice: "))
        print(MENU_DIVIDER)

        if choice == 7:
            break
        elif choice == 6:
            chosen_heuristic = run_search_algorithm(choice, test_maze, start, end, None)
            continue

        start_time = time.perf_counter()  # Measure start time
        path, iterations = run_search_algorithm(choice, test_maze, start, end, chosen_heuristic)
        end_time = time.perf_counter()  # Measure end time
        
        if path:
            # Copy original maze
            display_maze = []
            for i in range(len(test_maze)):
                display_maze.append([])
                for j in range(len(test_maze[i])):
                    display_maze[i].append(test_maze[i][j])
            # print("Path:", path)
            print("Path lenght:", len(path))
            print("Iterations:", iterations)
            k = 20
            for step_i, step_j in path:
                if display_maze[step_i][step_j] == PATH:
                    display_maze[step_i][step_j] = k
            display_maze[start[0]][start[1]] = 10
            display_maze[end[0]][end[1]] = 50
            for i in range(len(display_maze)):
                for j in range(len(display_maze[i])):
                    if display_maze[i][j] == WALL:
                        display_maze[i][j] = 6
            # Calculate and print elapsed time
            elapsed_time = end_time - start_time
            print("Elapsed time:", elapsed_time * 1000, "milliseconds")            
            # print_matrix(test_maze)
            np_array = np.array(display_maze)
            plt.matshow(np_array)
            plt.xticks([]) # remove the tick marks by setting to an empty list
            plt.yticks([]) # remove the tick marks by setting to an empty list
            plt.show()

if __name__ == "__main__":
    main()
