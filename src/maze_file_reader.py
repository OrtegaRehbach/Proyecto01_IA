class MazeFileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.maze = self._read_maze_from_file()

    def _read_maze_from_file(self):
        with open(self.file_path, "r") as file:
            maze_lines = file.readlines()

        maze = []
        for line in maze_lines:
            line = line.strip()  # Remove leading and trailing whitespace characters
            row = [int(char) for char in line]  # Convert each character to integer
            maze.append(row)
        return maze

    def get_maze(self):
        return self.maze

    def print_maze(self):
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                print( str(maze[i][j]).ljust(2),end=' ')
        print()

# Example usage:
# reader = MazeFileReader("maze.txt")
# maze = reader.get_maze()
# reader.print_maze()
if __name__ == "__main__":
    reader = MazeFileReader("./res/test_maze.txt")
    maze = reader.get_maze()
    reader.print_maze()