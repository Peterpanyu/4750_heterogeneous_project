import numpy as np
import math
import time
import sys
import matplotlib.pyplot as plt
from functools import wraps
from time import time
import contextlib
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from maze import generate_maze

#import relevant pycuda modules
kernel_code = """
__global__ void dijkstra(int *graph, int *dist, int *visited, int vertices, int src, float smoothing_factor) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= vertices) return;

    if (tid == src) {
        dist[tid] = 0;
    }

    __syncthreads();

    for (int step = 0; step < vertices; ++step) {
        int minDist = INT_MAX;
        int u = -1;

        if (tid == 0) {
            // Find the vertex with the smallest distance
            for (int i = 0; i < vertices; ++i) {
                if (!visited[i] && dist[i] < minDist) {
                    minDist = dist[i];
                    u = i;
                }
            }
            if (u != -1) {
                visited[u] = 1;
            }
        }

        __syncthreads();

        if (u != -1 && !visited[tid] && graph[u * vertices + tid] != 0) {
            int newDist = dist[u] + graph[u * vertices + tid];
            newDist = (int)(newDist * smoothing_factor); // Apply smoothing
            if (newDist < dist[tid]) {
                dist[tid] = newDist;
            }
        }

        __syncthreads();
    }
}
"""

# Initialize graph and parameters
def dijkstra_gpu(graph, src, smoothing_factor = 1.0):
    start = cuda.Event()
    end = cuda.Event()
    vertices = len(graph)
    graph_flatten = np.array(graph, dtype=np.int32).flatten()
    dist = np.full(vertices, np.iinfo(np.int32).max, dtype=np.int32)
    visited = np.zeros(vertices, dtype=np.int32)

    # Allocate GPU memory
    graph_gpu = cuda.mem_alloc(graph_flatten.nbytes)
    dist_gpu = cuda.mem_alloc(dist.nbytes)
    visited_gpu = cuda.mem_alloc(visited.nbytes)

    # Copy data to GPU
    cuda.memcpy_htod(graph_gpu, graph_flatten)
    cuda.memcpy_htod(dist_gpu, dist)
    cuda.memcpy_htod(visited_gpu, visited)

    # Compile kernel
    start.record()
    mod = SourceModule(kernel_code)
    dijkstra = mod.get_function("dijkstra")

    # Launch kernel
    block_size = 256
    grid_size = (vertices + block_size - 1) // block_size
    dijkstra(graph_gpu, dist_gpu, visited_gpu, np.int32(vertices), np.int32(src),np.float32(smoothing_factor),
             block=(block_size, 1, 1), grid=(grid_size, 1))
    end.record()
    end.synchronize()

    # Copy results back to host
    cuda.memcpy_dtoh(dist, dist_gpu)

    # Free GPU memory
    graph_gpu.free()
    dist_gpu.free()
    visited_gpu.free()

    return dist,start.time_till(end)


def dijkstra(graph, src,smoothing_factor = 1.0):
    """
    Find the shortest paths from src to all other vertices in a graph.
    
    Parameters:
        graph (list of lists): Adjacency matrix of the graph where
                               graph[i][j] represents the weight of the edge from i to j.
                               Use 0 for no connection.
        src (int): Index of the source vertex.
    
    Returns:
        list: Shortest distances from the source vertex to each vertex.
    """
    vertices = len(graph)
    dist = [sys.maxsize] * vertices  # Initialize distances to infinity
    visited = [False] * vertices    # Keep track of visited vertices

    dist[src] = 0  # Distance to source is 0
    ts = time()
    for _ in range(vertices):
        # Find the vertex with the minimum distance that hasn't been visited
        min_dist = sys.maxsize
        u = -1

        for i in range(vertices):
            if not visited[i] and dist[i] < min_dist:
                min_dist = dist[i]
                u = i

        if u == -1:  # If no unvisited vertex is reachable, break
            break

        # Mark the chosen vertex as visited
        visited[u] = True

        # Update the distance values of the adjacent vertices
        for v in range(vertices):
            if graph[u][v] > 0 and not visited[v]:
                new_dist = dist[u] + graph[u][v]
                new_dist = int(new_dist * smoothing_factor)  # Apply smoothing
                if new_dist < dist[v]:
                    dist[v] = new_dist
    te = time()
    return dist,(te - ts)*1000

# Example usage
if __name__ == "__main__":
    
    total_average_GPU_time=0
    total_average_CPU_time=0
    n = 100
    m = 10
    size = [20,30,40,50,100,500,2500]
    gpu_time = []
    gpu_improve = []
    cpu_time = []
    speedup = []

    for num_size in size:

        for i in range(n):
            total_GPU_time = 0
            total_CPU_time = 0
            # num_vertices = 1000  # Adjust for larger or smaller graphs
            # print(f"Graph number {i}:")
            # print("Generating graphs...")
            # print(f"number of vertices: {num_vertices}")
            graph = generate_maze(num_size)
            src = 0
            # print("Graph generated.")
            for j in range(m):
                # print(f"iteration {j}:")
                shortest_distances , GPU_time = dijkstra_gpu(graph, src)
                # print(f"GPU time is: {GPU_time}")
                shortest_distances_cpu ,CPU_time = dijkstra(graph,src)
                # print(f"CPU time is: {CPU_time}")
                total_GPU_time += GPU_time
                total_CPU_time += CPU_time
            average_GPU_time = total_GPU_time/m
            average_CPU_time = total_CPU_time/m
            
            # print(f"average GPU time is {average_GPU_time}ms")
            # print(f"average CPU time is {average_CPU_time}ms")
            # print(f"average improement is {(average_CPU_time-average_GPU_time)/average_CPU_time*100}%")

            total_average_GPU_time += average_GPU_time
            total_average_CPU_time += average_CPU_time
        total_average_GPU_time = total_average_GPU_time/n
        total_average_CPU_time = total_average_CPU_time/n
        improvement = (total_average_CPU_time-total_average_GPU_time)/total_average_CPU_time*100
        print(f"The average improvement with {num_size}*{num_size} maze: {improvement}%")
        print(f"The speedup factor is {total_average_CPU_time/total_average_GPU_time}")
        gpu_time.append(total_average_GPU_time)
        cpu_time.append(total_average_CPU_time)
        gpu_improve.append(improvement)
        speedup.append(total_average_CPU_time/total_average_GPU_time)

    print("Maze size:",size)
    print("GPU average time:",gpu_time)
    print("CPU average time:",cpu_time)
    print("Improvement",gpu_improve)
    print("Speedup Factors:",speedup)
