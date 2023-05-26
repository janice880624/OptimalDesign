import numpy as np
import random
import matplotlib.pyplot as plt

# 定義問題參數
start_point = (0, 0)  # 起點座標
end_point = (100, 100)  # 終點座標
num_waypoints = 5  # 中繼站數量

best_waypoints = []

start_end =[start_point, end_point]

x1 = [point[0] for point in start_end]
y1 = [point[1] for point in start_end]

plt.scatter(x1, y1)

# 遺傳演算法參數
population_size = 100  # 種群大小
num_generations = 20  # 演化代數
mutation_rate = 0.01  # 突變率

# 生成隨機中繼站座標
waypoints = []
for _ in range(num_waypoints):
    x = random.uniform(start_point[0], end_point[0])
    y = random.uniform(start_point[1], end_point[1])
    waypoints.append((x, y))


print("中繼站: ", waypoints)
# Extract x and y coordinates
x2 = [point[0] for point in waypoints]
y2 = [point[1] for point in waypoints]

# Create scatter plot
plt.scatter(x2, y2)

# Set plot title and labels
plt.title("Scatter Plot")
plt.xlabel("X")
plt.ylabel("Y")

# Show the plot
# plt.show()

# 定義染色體編碼
# 每個染色體代表一條路徑，包括起點、中繼站和終點
chromosome_length = num_waypoints + 2

# 計算兩點之間的歐氏距離
def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# 計算染色體的適應度，即路徑總長度
def calculate_fitness(chromosome):
    total_distance = distance(start_point, chromosome[0])
    for i in range(len(chromosome) - 1):
        total_distance += distance(chromosome[i], chromosome[i+1])
    total_distance += distance(chromosome[-1], end_point)
    return total_distance

# 初始化種群
population = []
for _ in range(population_size):
    chromosome = [start_point] + random.sample(waypoints, num_waypoints) + [end_point]
    population.append(chromosome)

# 主要演化循環
for generation in range(num_generations):
    # 計算每個染色體的適應度
    fitness_values = [calculate_fitness(chromosome) for chromosome in population]

    # 選擇父代染色體
    mating_pool = []
    for _ in range(population_size):
        # 使用競爭性選擇方法，選擇兩個隨機染色體進行比較
        random_indices = random.sample(range(population_size), 2)
        chromosome1 = population[random_indices[0]]
        chromosome2 = population[random_indices[1]]
        fitness1 = fitness_values[random_indices[0]]
        fitness2 = fitness_values[random_indices[1]]
        if fitness1 < fitness2:
            mating_pool.append(chromosome1)
        else:
            mating_pool.append(chromosome2)

    # 交配產生子代染色體
    offspring_population = []

    while len(offspring_population) < population_size:
        # 選擇兩個父代染色體進行交配
        parent1 = random.choice(mating_pool)
        parent2 = random.choice(mating_pool)
        
        # 進行交叉操作
        crossover_point = random.randint(1, chromosome_length - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        # 進行突變操作
        if random.random() < mutation_rate:
            mutation_point = random.randint(1, chromosome_length - 2)
            child1[mutation_point] = random.choice(waypoints)
        if random.random() < mutation_rate:
            mutation_point = random.randint(1, chromosome_length - 2)
            child2[mutation_point] = random.choice(waypoints)
        
        # 將子代染色體加入子代種群
        offspring_population.append(child1)
        offspring_population.append(child2)

    # 更新種群
    population = offspring_population
    
    # 計算當前最佳解和最佳適應度
    best_chromosome = min(population, key=calculate_fitness)
    best_fitness = calculate_fitness(best_chromosome)

    print("Best solution:")
    for i, point in enumerate(best_chromosome):
        print(f"{i+1}. Position: {point}")

    print(f"Best Fitness: {best_fitness}")
        
    # 輸出每一代的最佳解和最佳適應度
    # print(f"Generation {generation+1}: Best Fitness = {best_fitness}")

print(population[-1])

x3 = [point[0] for point in mating_pool[0]]
y3 = [point[1] for point in mating_pool[0]]

print(x3)
print(y3)

plt.plot(x3, y3, 'bo-')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Connected Points')

plt.show()

print(f"Generation {generation+1}: Best Fitness = {best_fitness}")
