import math
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

start = (0, 0, 0)
end = (100, 100, 0)
intermediates = [(50, 35, 90), (95, 83, 78), (31, 63, 20), (17, 54, 76), (69, 18, 34), (47, 74, 19), (95, 36, 73), (15, 43, 58), (58, 56, 68), (87, 23, 97)]

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def total_distance(path):
    return sum(calculate_distance(path[i], path[i+1]) for i in range(len(path)-1))

def greedy_shortest_path(start, intermediates, end):
    path = [start]
    while intermediates:
        next_point = min(intermediates, key=lambda x: calculate_distance(path[-1], x))
        intermediates.remove(next_point)
        path.append(next_point)
    path.append(end)
    return path

def main():
    start_time = time.time()
    best_path = greedy_shortest_path(start, intermediates, end)
    execution_time = time.time() - start_time

    print("Best Path:", best_path)
    print("Path Length:", total_distance(best_path))
    print("Execution Time:", execution_time, "s")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(start[0], start[1], start[2], c='r', marker='o', label='Start')
    ax.scatter(end[0], end[1], end[2], c='orange', marker='o', label='End')

    for i, point in enumerate(intermediates):
        ax.scatter(point[0], point[1], point[2], c='b', marker='o')
        ax.text(point[0], point[1], point[2], str(i), color='b', fontsize=10, verticalalignment='bottom')

    x = [point[0] for point in best_path]
    y = [point[1] for point in best_path]
    z = [point[2] for point in best_path]
    ax.plot(x, y, z, marker='o', label='Best Path')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
