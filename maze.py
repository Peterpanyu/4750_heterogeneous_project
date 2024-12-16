#Used randomized depth-first search (DFS) algorithm to generate the large size maze
#Yu Pan
#yp2742

import random

def generate_maze(size):
    height = size
    width = size
    """
    Generate a maze with a guaranteed path from the top-left corner to the bottom-right corner.

    :param width: Number of columns in the maze
    :param height: Number of rows in the maze
    :return: A 2D list representing the maze (0 for walls, 1 for paths)
    """
    # Create a grid filled with walls (0)
    maze = [[0 for _ in range(width)] for _ in range(height)]

    # Directions for moving in the maze: (row offset, column offset)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Iterative Depth-First Search to carve the maze
    def carve_passages_iterative(x, y):
        stack = [(x, y)]  # Use a stack to avoid recursion limit
        visited = set()
        visited.add((x, y))
        while stack:
            cx, cy = stack[-1]  # Look at the top of the stack
            maze[cy][cx] = 1  # Mark the current cell as a passage
            random.shuffle(directions)  # Randomize directions
            next_moves = []
            for dx, dy in directions:
                nx, ny = cx + dx * 2, cy + dy * 2
                if 0 <= ny < height and 0 <= nx < width and (nx, ny) not in visited:
                    next_moves.append((nx, ny, cx + dx, cy + dy))

            if next_moves:
                nx, ny, wall_x, wall_y = random.choice(next_moves)
                maze[wall_y][wall_x] = 1  # Carve through the wall
                stack.append((nx, ny))
                visited.add((nx, ny))
            else:
                stack.pop()  # Backtrack if no moves are available

    # Start carving from the top-left corner
    carve_passages_iterative(0, 0)

    # Ensure the bottom-right corner is part of the path
    maze[height - 1][width - 1] = 1

    return maze

# # Display the maze
# def print_maze(maze):
#     """Print the maze in a readable format."""
#     for row in maze:
#         print("".join(" █"[cell] for cell in row))

# # Main function
# if __name__ == "__main__":
#     import sys

#     # Allow dynamic input for large maze sizes
#     try:
#         size = int(input("Enter maze size (must be odd): "))
#         if size % 2 == 0 :
#             print("Size must be odd numbers.")
#             sys.exit(1)
#     except ValueError:
#         print("Invalid input. Please enter integer values.")
#         sys.exit(1)

#     maze = generate_maze(size)
#     print_maze(maze)