import numpy as np
import time
import sys
from time import time
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from maze import generate_maze
from heapq import heappop, heappush

#import relevant pycuda modules
kernel_code = """
__global__ void dijkstra_with_goal(int *graph, int *dist, int *visited, int vertices, int goal, int *path_found) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= vertices || *path_found) return;

    for (int i = 0; i < vertices; ++i) {
        int min_dist = INT_MAX;
        int u = -1;

        // Thread 0 finds the next closest unvisited vertex
        if (tid == 0) {
            for (int v = 0; v < vertices; ++v) {
                if (!visited[v] && dist[v] < min_dist) {
                    min_dist = dist[v];
                    u = v;
                }
            }
            if (u == goal) {
                *path_found = 1; // Mark path found if goal is reached
            }
            if (u != -1) visited[u] = 1;
        }
        __syncthreads();

        // Update distances for the current vertex
        if (u != -1 && tid < vertices && graph[u * vertices + tid] > 0) {
            int new_dist = dist[u] + graph[u * vertices + tid];
            if (new_dist < dist[tid]) {
                dist[tid] = new_dist;
            }
        }
        __syncthreads();
    }
}

__device__ int heuristic(int x1, int y1, int x2, int y2) {
    // Manhattan distance heuristic
    return abs(x1 - x2) + abs(y1 - y2);
}

__global__ void a_star(int *maze, int *g_score, int *f_score, int *visited, 
                       int width, int height, int goal_x, int goal_y, int *path_found) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int x = tid / width;
    int y = tid % width;

    if (x >= height || y >= width) return;

    // Stop if goal is found
    if (*path_found) return;

    __shared__ int next_node_x, next_node_y, min_f;

    if (tid == 0) {
        min_f = INT_MAX;
    }

    __syncthreads();

    if (!visited[tid] && f_score[tid] < min_f) {
        atomicMin(&min_f, f_score[tid]);
        atomicExch(&next_node_x, x);
        atomicExch(&next_node_y, y);
    }

    __syncthreads();

    // Expand the node with the smallest f_score
    if (x == next_node_x && y == next_node_y && !visited[tid]) {
        visited[tid] = 1; // Mark node as visited

        // Explore neighbors
        int neighbors[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int i = 0; i < 4; i++) {
            int nx = x + neighbors[i][0];
            int ny = y + neighbors[i][1];

            if (nx >= 0 && nx < height && ny >= 0 && ny < width && maze[nx * width + ny] == 0) {
                int neighbor_id = nx * width + ny;

                int tentative_g = g_score[tid] + 1;
                if (tentative_g < g_score[neighbor_id]) {
                    g_score[neighbor_id] = tentative_g;
                    f_score[neighbor_id] = tentative_g + heuristic(nx, ny, goal_x, goal_y);

                    // If the goal is reached
                    if (nx == goal_x && ny == goal_y) {
                        *path_found = 1;
                    }
                }
            }
        }
    }
}
"""

# Initialize graph and parameters
def dijkstra_gpu(graph, start, goal):
    start_event = cuda.Event()
    end_event = cuda.Event()

    vertices = len(graph)
    graph_flatten = np.array(graph, dtype=np.int32).flatten()
    dist = np.full(vertices, np.iinfo(np.int32).max, dtype=np.int32)
    visited = np.zeros(vertices, dtype=np.int32)
    path_found = np.array([0], dtype=np.int32)

    start_idx = start[0] * len(graph[0]) + start[1]
    goal_idx = goal[0] * len(graph[0]) + goal[1]
    dist[start_idx] = 0

    start_event.record()

    # Allocate GPU memory
    graph_gpu = cuda.mem_alloc(graph_flatten.nbytes)
    dist_gpu = cuda.mem_alloc(dist.nbytes)
    visited_gpu = cuda.mem_alloc(visited.nbytes)
    path_found_gpu = cuda.mem_alloc(path_found.nbytes)

    # Copy data to GPU
    cuda.memcpy_htod(graph_gpu, graph_flatten)
    cuda.memcpy_htod(dist_gpu, dist)
    cuda.memcpy_htod(visited_gpu, visited)
    cuda.memcpy_htod(path_found_gpu, path_found)

    # Compile kernel
    mod = SourceModule(kernel_code)
    dijkstra = mod.get_function("dijkstra_with_goal")

    # Launch kernel
    block_size = 256
    grid_size = (vertices + block_size - 1) // block_size
    dijkstra(graph_gpu, dist_gpu, visited_gpu, np.int32(vertices), np.int32(goal_idx), path_found_gpu,
             block=(block_size, 1, 1), grid=(grid_size, 1))

    # Copy results back to host
    cuda.memcpy_dtoh(dist, dist_gpu)
    cuda.memcpy_dtoh(path_found, path_found_gpu)

    end_event.record()
    end_event.synchronize()

    # Free GPU memory
    graph_gpu.free()
    dist_gpu.free()
    visited_gpu.free()
    path_found_gpu.free()

    return dist, start_event.time_till(end_event)


def dijkstra_with_goal(maze, start, goal):
    """
    Modified Dijkstra's algorithm to stop when the goal is reached.
    :param maze: 2D list representing the maze
    :param start: Tuple (x, y) for the start position
    :param goal: Tuple (x, y) for the goal position
    :return: Distance to the goal and path found status
    """
    import heapq
    ts = time()
    height, width = len(maze), len(maze[0])
    distances = [[float('inf')] * width for _ in range(height)]
    visited = [[False] * width for _ in range(height)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    sx, sy = start
    gx, gy = goal
    distances[sy][sx] = 0
    queue = [(0, sx, sy)]  # Min-heap with (distance, x, y)

    while queue:
        dist, x, y = heapq.heappop(queue)

        if (x, y) == goal:
            te = time()
            return dist, (te-ts)*1000  # Return the distance when the goal is reached

        if visited[y][x]:
            continue
        visited[y][x] = True

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 1 and not visited[ny][nx]:
                new_dist = dist + 1
                if new_dist < distances[ny][nx]:
                    distances[ny][nx] = new_dist
                    heapq.heappush(queue, (new_dist, nx, ny))

    return float('inf'), False

def heuristic(x1, y1, x2, y2):
    """Calculate the Manhattan distance between two points (x1, y1) and (x2, y2)."""
    return abs(x1 - x2) + abs(y1 - y2)

def a_star_search(maze, start, goal):
    """
    Perform A* search on a maze.
    Returns:
        bool: Whether a path from start to goal is found.
        float: The execution time in milliseconds.
    """
    height, width = len(maze), len(maze[0])
    open_set = []  # Priority queue for open nodes
    heappush(open_set, (0, start))  # (priority, node)

    came_from = {}  # Map of navigated nodes
    g_score = {start: 0}  # Cost from start to node
    f_score = {start: heuristic(start[0], start[1], goal[0], goal[1])}  # Estimated cost to goal

    start_time = time()  # Start timing

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            end_time = time()  # End timing
            return True, (end_time - start_time) * 1000

        x, y = current
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

        for nx, ny in neighbors:
            if 0 <= nx < height and 0 <= ny < width and maze[nx][ny] == 0:
                tentative_g_score = g_score[current] + 1
                if (nx, ny) not in g_score or tentative_g_score < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g_score
                    f_score[(nx, ny)] = tentative_g_score + heuristic(nx, ny, goal[0], goal[1])
                    heappush(open_set, (f_score[(nx, ny)], (nx, ny)))

    end_time = time()  # End timing
    return False, (end_time - start_time) * 1000

def a_star_gpu(maze, start, goal):
    start_event = cuda.Event()
    end_event = cuda.Event()

    height, width = len(maze), len(maze[0])
    maze_flat = np.array(maze, dtype=np.int32).flatten()

    g_score = np.full(height * width, np.iinfo(np.int32).max, dtype=np.int32)
    f_score = np.full(height * width, np.iinfo(np.int32).max, dtype=np.int32)
    visited = np.zeros(height * width, dtype=np.int32)
    path_found = np.array([0], dtype=np.int32)

    start_idx = start[0] * width + start[1]
    goal_x, goal_y = goal

    g_score[start_idx] = 0
    f_score[start_idx] = heuristic(start[0], start[1], goal[0], goal[1])

    # Start timing
    start_event.record()

    # Allocate GPU memory
    maze_gpu = cuda.mem_alloc(maze_flat.nbytes)
    g_score_gpu = cuda.mem_alloc(g_score.nbytes)
    f_score_gpu = cuda.mem_alloc(f_score.nbytes)
    visited_gpu = cuda.mem_alloc(visited.nbytes)
    path_found_gpu = cuda.mem_alloc(path_found.nbytes)

    # Copy data to GPU
    cuda.memcpy_htod(maze_gpu, maze_flat)
    cuda.memcpy_htod(g_score_gpu, g_score)
    cuda.memcpy_htod(f_score_gpu, f_score)
    cuda.memcpy_htod(visited_gpu, visited)
    cuda.memcpy_htod(path_found_gpu, path_found)

    # Compile and launch kernel
    mod = SourceModule(kernel_code)
    a_star = mod.get_function("a_star")

    block_size = 256
    grid_size = (height * width + block_size - 1) // block_size
    a_star(maze_gpu, g_score_gpu, f_score_gpu, visited_gpu, np.int32(width), np.int32(height),
           np.int32(goal_x), np.int32(goal_y), path_found_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))

    # Copy results back to host
    cuda.memcpy_dtoh(path_found, path_found_gpu)

    # End timing
    end_event.record()
    end_event.synchronize()

    # Free GPU memory
    maze_gpu.free()
    g_score_gpu.free()
    f_score_gpu.free()
    visited_gpu.free()
    path_found_gpu.free()

    return bool(path_found[0]), start_event.time_till(end_event)

# Example usage
if __name__ == "__main__":
    total_average_GPU_dijkstra_time = 0
    total_average_CPU_dijkstra_time = 0
    total_average_GPU_a_star_time = 0
    total_average_CPU_a_star_time = 0

    n = 50  # Number of mazes per size
    m = 3  # Number of runs per maze
    sizes = [21, 31, 41, 51, 101,501,2501]  # Maze sizes
    gpu_dijkstra_times = []
    cpu_dijkstra_times = []
    gpu_a_star_times = []
    cpu_a_star_times = []
    speedup_dijkstra = []
    speedup_a_star = []

    for maze_size in sizes:
        print(f"Benchmarking for maze size: {maze_size}x{maze_size}")
        total_GPU_dijkstra_time = total_CPU_dijkstra_time = 0
        total_GPU_a_star_time = total_CPU_a_star_time = 0

        for _ in range(n):
            maze = generate_maze(maze_size)
            start = (0, 0)
            goal = (maze_size - 1, maze_size - 1)

            for _ in range(m):
                # GPU Dijkstra
                _, GPU_dijkstra_time = dijkstra_gpu(maze, start, goal)
                total_GPU_dijkstra_time += GPU_dijkstra_time

                # CPU Dijkstra
                _, CPU_dijkstra_time = dijkstra_with_goal(maze, start, goal)
                total_CPU_dijkstra_time += CPU_dijkstra_time

                # GPU A*
                _, GPU_a_star_time = a_star_gpu(maze, start, goal)
                total_GPU_a_star_time += GPU_a_star_time

                # CPU A*
                _, CPU_a_star_time = a_star_search(maze, start, goal)
                total_CPU_a_star_time += CPU_a_star_time

        # Average times
        avg_GPU_dijkstra_time = total_GPU_dijkstra_time / (n * m)
        avg_CPU_dijkstra_time = total_CPU_dijkstra_time / (n * m)
        avg_GPU_a_star_time = total_GPU_a_star_time / (n * m)
        avg_CPU_a_star_time = total_CPU_a_star_time / (n * m)

        # Store results
        gpu_dijkstra_times.append(avg_GPU_dijkstra_time)
        cpu_dijkstra_times.append(avg_CPU_dijkstra_time)
        gpu_a_star_times.append(avg_GPU_a_star_time)
        cpu_a_star_times.append(avg_CPU_a_star_time)
        speedup_dijkstra.append(avg_CPU_dijkstra_time / avg_GPU_dijkstra_time)
        speedup_a_star.append(avg_CPU_a_star_time / avg_GPU_a_star_time)

        # Print results for this maze size
        print(f"Maze size: {maze_size}x{maze_size}")
        print(f"  Dijkstra GPU Time: {avg_GPU_dijkstra_time:.2f} ms")
        print(f"  Dijkstra CPU Time: {avg_CPU_dijkstra_time:.2f} ms")
        print(f"  Dijkstra Speedup: {avg_CPU_dijkstra_time / avg_GPU_dijkstra_time:.2f}x")
        print(f"  A* GPU Time: {avg_GPU_a_star_time:.2f} ms")
        print(f"  A* CPU Time: {avg_CPU_a_star_time:.2f} ms")
        print(f"  A* Speedup: {avg_CPU_a_star_time / avg_GPU_a_star_time:.2f}x")

    # Print summary
    print("Maze Sizes:", sizes)
    print("GPU Dijkstra Times (ms):", gpu_dijkstra_times)
    print("CPU Dijkstra Times (ms):", cpu_dijkstra_times)
    print("GPU A* Times (ms):", gpu_a_star_times)
    print("CPU A* Times (ms):", cpu_a_star_times)
    print("Dijkstra Speedup:", speedup_dijkstra)
    print("A* Speedup:", speedup_a_star)
