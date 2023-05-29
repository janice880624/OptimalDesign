import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define middle points
middle_points = [(50, 35, 90), (95, 83, 78), (31, 63, 20), (17, 54, 76), (69, 18, 34),
                 (47, 74, 19), (95, 36, 73), (15, 43, 58), (58, 56, 68), (87, 23, 97)]

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)

def calculate_fitness(path):
    fitness = 0
    prev_point = (0, 0, 0)
    for point in path:
        fitness += calculate_distance(prev_point, point)
        prev_point = point
    fitness += calculate_distance(prev_point, (100, 100, 0))
    return fitness

def generate_random_path():
    path = random.sample(middle_points, len(middle_points))
    return path

def crossover(parent1, parent2):
    offspring1 = []
    offspring2 = []
    for point1, point2 in zip(parent1, parent2):
        if point1 in parent2:
            offspring1.append(point1)
        else:
            offspring1.append(None)
        if point2 in parent1:
            offspring2.append(point2)
        else:
            offspring2.append(None)

    for i in range(len(offspring1)):
        if offspring1[i] is None:
            offspring1[i] = parent2[i]
        if offspring2[i] is None:
            offspring2[i] = parent1[i]

    return offspring1, offspring2

def mutate(path, mutation_rate):
    for i in range(len(path)):
        if random.random() < mutation_rate:
            swap_index = random.randint(0, len(middle_points) - 1)
            path[i], path[swap_index] = path[swap_index], path[i]

def genetic_algorithm(population_size, num_generations, mutation_rate):
    population = [generate_random_path() for _ in range(population_size)]
    best_fitnesses = []
    best_individuals = []

    for generation in range(num_generations):
        fitness_values = [calculate_fitness(path) for path in population]
        min_fitness = min(fitness_values)
        best_individual = population[fitness_values.index(min_fitness)]

        best_fitnesses.append(min_fitness)
        best_individuals.append(best_individual)

        new_population = [best_individual]

        while len(new_population) < population_size:
            parents = random.choices(population, weights=fitness_values, k=2)
            offspring1, offspring2 = crossover(parents[0], parents[1])
            mutate(offspring1, mutation_rate)
            mutate(offspring2, mutation_rate)
            new_population.extend([offspring1, offspring2])

        population = new_population

        print("Generation:", generation, " Best Fitness:", min_fitness)

    best_individual = population[fitness_values.index(min_fitness)]
    best_fitnesses.append(min_fitness)
    best_individuals.append(best_individual)

    return best_individuals, best_fitnesses

# Run the genetic algorithm
population_size = 100
num_generations = 10000
mutation_rate = 0.1
start_time = time.time()
best_individuals, best_fitnesses = genetic_algorithm(population_size, num_generations, mutation_rate)

# Extract the best path
best_path = best_individuals[-1]
best_path = [(0, 0, 0)] + best_path + [(100, 100, 0)]

# Calculate execution time
execution_time = time.time() - start_time
# Print best path, execution time, and path length
print("Best Path:", best_path)
print("Execution Time:", execution_time)
print("Path Length:", best_fitnesses[-1])

# Plotting 3D path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



x = [point[0] for point in best_path]
y = [point[1] for point in best_path]
z = [point[2] for point in best_path]
ax.scatter(x[0], y[0], z[0], c='red', marker='o', label='Start')
ax.scatter(x[-1], y[-1], z[-1], c='orange', marker='o', label='End')
ax.plot(x, y, z, c='b', marker='o', label='Best Path')
ax.plot(x, y, z, marker='o')



ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()