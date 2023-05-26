import time
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 記錄開始時間
start_time = time.time()

start_point = (0, 0, 0)  
end_point = (100, 100, 0)  
num_waypoints = 10  

waypoints = []
best_waypoints = []
population = []
offspring_population = []

# GA演算法參數
population_size = 100
num_generations = 30000  
mutation_rate = 0.00000001 

start_end =[start_point, end_point]
x1 = [point[0] for point in start_end]
y1 = [point[1] for point in start_end]
z1 = [point[2] for point in start_end]

# 隨機生成中繼站座標
for _ in range(num_waypoints):
    x = random.uniform(start_point[0], end_point[0])
    y = random.uniform(start_point[1], end_point[1])
    z = random.uniform(start_point[2], 100)
    waypoints.append((x, y, z))

chromosome_length = num_waypoints + 2

def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)

def calculate_fitness(chromosome):
    total_distance = distance(start_point, chromosome[0])
    for i in range(len(chromosome) - 1):
        total_distance += distance(chromosome[i], chromosome[i+1])
    total_distance += distance(chromosome[-1], end_point)
    return total_distance

for _ in range(population_size):
    chromosome = [start_point] + random.sample(waypoints, num_waypoints) + [end_point]
    population.append(chromosome)

for generation in range(num_generations):
    fitness_values = [calculate_fitness(chromosome) for chromosome in population]

    mating_pool = []
    for _ in range(population_size):
        random_indices = random.sample(range(population_size), 2)
        chromosome1 = population[random_indices[0]]
        chromosome2 = population[random_indices[1]]
        fitness1 = fitness_values[random_indices[0]]
        fitness2 = fitness_values[random_indices[1]]
        if fitness1 < fitness2:
            mating_pool.append(chromosome1)
        else:
            mating_pool.append(chromosome2)

    while len(offspring_population) < population_size:
        parent1 = random.choice(mating_pool)
        parent2 = random.choice(mating_pool)
        
        crossover_point = random.randint(1, chromosome_length - 1)

        child1 = parent1[:crossover_point]
        child2 = parent2[:crossover_point]

        for point in parent2:
            if point not in child1:
                child1.append(point)
        for point in parent1:
            if point not in child2:
                child2.append(point)
        
        if random.random() < mutation_rate:
            mutation_point = random.randint(1, num_waypoints)
            child1[mutation_point] = waypoints[mutation_point - 1]
        if random.random() < mutation_rate:
            mutation_point = random.randint(1, num_waypoints)
            child2[mutation_point] = waypoints[mutation_point - 1]
        
        offspring_population.append(child1)
        offspring_population.append(child2)

    population = offspring_population
    
    # 計算當前最佳解和最佳適應度
    best_chromosome = min(population, key=calculate_fitness)
    best_fitness = calculate_fitness(best_chromosome)

    # print("Best solution:")
    # for i, point in enumerate(best_chromosome):
    #     print(f"{i+1}. Position: {point}")

    # print(f"Best Fitness: {best_fitness}")
        
x3 = [point[0] for point in population[-1]]
y3 = [point[1] for point in population[-1]]
z3 = [point[2] for point in population[-1]]

# 創建3D圖形物件
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x3, y3, z3, c='r', marker='o')
ax.plot(x3, y3, z3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

end_time = time.time()
execution_time = end_time - start_time
print("程式執行時間：", execution_time, "秒")

plt.show()

print(f"Generation {generation+1}: Best Fitness = {best_fitness}")

