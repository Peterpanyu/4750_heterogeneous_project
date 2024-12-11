import random
import matplotlib.pyplot as plt

def generate_maze(vertices):
    height = vertices
    width = vertices
    # Initialize the maze grid
    maze = [[1 for _ in range(width)] for _ in range(height)]
    
    # Starting position
    stack = [(0, 0)]  # Stack for backtracking
    maze[0][0] = 0  # Mark starting cell as a path

    # Possible movement directions (row_offset, col_offset)
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    
    while stack:
        current_cell = stack[-1]
        x, y = current_cell

        # Find all valid neighbors
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and maze[nx][ny] == 1:
                neighbors.append((nx, ny))
        
        if neighbors:
            # Choose a random neighbor
            nx, ny = random.choice(neighbors)
            
            # Remove the wall between the current cell and the chosen neighbor
            wall_x, wall_y = (x + nx) // 2, (y + ny) // 2
            maze[wall_x][wall_y] = 0
            
            # Mark the neighbor as part of the maze
            maze[nx][ny] = 0
            
            # Push the neighbor to the stack
            stack.append((nx, ny))
        else:
            # Backtrack if no valid neighbors
            stack.pop()

    return maze

def display_maze(maze):
    # Display the maze using Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap='binary')  # Use binary colormap (0=black, 1=white)
    plt.axis('off')
    plt.show()

# Generate and display a large maze
width, height = 51, 51  # Maze dimensions (must be odd)
maze = generate_maze(width, height)
display_maze(maze)