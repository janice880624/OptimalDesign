import numpy as np
import random
import time
from scipy.spatial import distance
import matplotlib.pyplot as plt

# 建立初始人口
def create_waypoints(size, num_waypoints):
    population = []
    for i in range(size):
        population.append(random.sample(range(1,num_waypoints-1),num_waypoints-2)) #不包含起點和終點
    return population

# 計算適應度
def compute_fitness(population, distance_matrix):
    fitness_scores = []
    for route in population:
        route = [0] + route + [len(distance_matrix)-1] # 加入固定的起點和終點
        score = 0
        for i in range(len(route) - 1):
            score += distance_matrix[route[i]][route[i+1]]
        fitness_scores.append(1/score) # 我們需要最小化路徑，所以適應度分數需要為距離的倒數
    return fitness_scores

# 選擇父母
def select_parents(fitness_scores):
    return np.random.choice(len(fitness_scores), 2, p=fitness_scores/np.sum(fitness_scores))

# 交配
def crossover(parent1, parent2):
    child = [-1]*len(parent1)
    geneA = int(random.random()*len(parent1))
    geneB = int(random.random()*len(parent1))
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    for i in range(startGene, endGene):
        child[i] = parent1[i]
    filled = endGene - startGene
    childIndex = endGene
    parentIndex = endGene
    while filled < len(parent1):
        if parentIndex >= len(parent1):
            parentIndex = 0
        if childIndex >= len(parent1):
            childIndex = 0
        if parent2[parentIndex] not in child:
            child[childIndex] = parent2[parentIndex]
            childIndex += 1
            filled += 1
        parentIndex += 1
    return child

# 突變
def mutate(route, mutation_rate):
    for swapped in range(len(route)):
        if(random.random() < mutation_rate):
            swap_with = int(random.random() * len(route))
            waypoints1 = route[swapped]
            waypoints2 = route[swap_with]
            route[swapped] = waypoints2
            route[swap_with] = waypoints1
    return route

# 遺傳演算法
def genetic_algorithm(waypoints_coordinates, pop_size=100, num_generations=1000, mutation_rate=0.01):
    
    start_time = time.time()
    num_waypoints = len(waypoints_coordinates)
    distance_matrix = np.zeros((num_waypoints, num_waypoints))
    for i in range(num_waypoints):
        for j in range(num_waypoints):
            distance_matrix[i,j] = distance.euclidean(waypoints_coordinates[i], waypoints_coordinates[j])

    population = create_waypoints(pop_size, num_waypoints)

    best_score = float('inf')
    best_route = []

    for generation in range(num_generations):
        fitness_scores = compute_fitness(population, distance_matrix)
        new_population = []
        for i in range(pop_size):
            parents = select_parents(fitness_scores)
            child = crossover(population[parents[0]], population[parents[1]])
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population
        fitness_scores = compute_fitness(population, distance_matrix)
        current_best_score_index = np.argmax(fitness_scores)
        current_best_score = 1/fitness_scores[current_best_score_index]
        if current_best_score < best_score:
            best_score = current_best_score
            best_route = [0] + population[current_best_score_index] + [num_waypoints-1]

        print(f"Generation {generation}: Best score: {current_best_score}") 
        print(f"Current best route: {[0] + population[current_best_score_index] + [num_waypoints-1]}")

    print("Final best score:", best_score)
    print("Final best route:", best_route)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    # 繪製最後的最佳路徑
    best_route_coordinates = [waypoints_coordinates[i] for i in best_route]
    plt.figure()
    plt.scatter(*zip(*waypoints_coordinates), c='b') # 所有城市
    plt.scatter(*zip(*[waypoints_coordinates[0], waypoints_coordinates[-1]]), c='r') # 起點和終點
    plt.plot(*zip(*best_route_coordinates), c='g') # 最佳路徑
    for i, txt in enumerate(best_route):
        plt.annotate(txt, (waypoints_coordinates[best_route[i]][0], waypoints_coordinates[best_route[i]][1]))
    plt.show()

def main():
    # 測試
    waypoints_coordinates = [(0, 0)]
    for _ in range(10):
        waypoints_coordinates.append((random.randint(1,99), random.randint(1,99)))
    waypoints_coordinates.append((100, 100))
    genetic_algorithm(waypoints_coordinates)


if __name__ == "__main__":
    main()
