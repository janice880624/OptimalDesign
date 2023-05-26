import time
import random
import numpy as np
import matplotlib.pyplot as plt

# 記錄開始時間
start_time = time.time()

start_point = (0, 0)  
end_point = (100, 100)  
num_waypoints = 5

waypoints = []
offspring_population = []

# GA 演算法參數
population_size = 100
num_generations = 100
mutation_rate = 0.01  

start_end =[start_point, end_point]
x1 = [point[0] for point in start_end]
y1 = [point[1] for point in start_end]

# 隨機生成中繼站座標
# for _ in range(num_waypoints):
#     x = random.uniform(start_point[0], end_point[0])
#     y = random.uniform(start_point[1], end_point[1])
#     waypoints.append((x, y))
waypoints = ((10, 10), (20, 20), (30, 30), (40, 40), (60, 80))

print("隨機產生的座標:{}".format(waypoints))

# 計算兩點的距離
def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 )

# 計算路徑長度(染色體的適應程度)
def calculate_fitness(chromosome):
    total_distance = distance(chromosome[0], chromosome[-1])
    for i in range(len(chromosome) - 1):
        total_distance += distance(chromosome[i], chromosome[i+1])
    return total_distance

def main():
    # 初始化種群
    population = []

    for _ in range(population_size):
        chromosome = random.sample(waypoints, num_waypoints)
        population.append(chromosome)

    best_fitness = float('inf')
    best_chromosome = []

    for generation in range(num_generations):
        fitness_values = [calculate_fitness(chromosome) for chromosome in population]

        mating_pool = []
        for _ in range(population_size):
            fitness_sum = sum(fitness_values)
            selection_probs = [fitness / fitness_sum for fitness in fitness_values]
            parent1 = random.choices(population, weights=selection_probs)[0]
            parent2 = random.choices(population, weights=selection_probs)[0]
            mating_pool.append((parent1, parent2))

        while len(offspring_population) < population_size:
            parents = random.choice(mating_pool)
            parent1 = parents[0]
            parent2 = parents[1]

            crossover_point = random.randint(1, num_waypoints - 1)

            child1 = parent1[:crossover_point]
            child2 = parent2[:crossover_point]

            for point in parent2:
                if point not in child1:
                    child1.append(point)
            for point in parent1:
                if point not in child2:
                    child2.append(point)
            
            if random.random() < mutation_rate:
                mutation_point1, mutation_point2 = random.sample(range(num_waypoints), 2)
                child1[mutation_point1], child1[mutation_point2] = child1[mutation_point2], child1[mutation_point1]
            if random.random() < mutation_rate:
                mutation_point1, mutation_point2 = random.sample(range(num_waypoints), 2)
                child2[mutation_point1], child2[mutation_point2] = child2[mutation_point2], child2[mutation_point1]
            
            offspring_population.append(child1)
            offspring_population.append(child2)

        population = offspring_population
        
        # 計算當前最佳解和最佳適應度
        current_best_chromosome = min(population, key=calculate_fitness)
        current_best_fitness = calculate_fitness(current_best_chromosome)

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = current_best_chromosome

    # 組合最佳解
    best_solution = [start_point] + best_chromosome + [end_point]
    x3 = [point[0] for point in best_solution]
    y3 = [point[1] for point in best_solution]

    plt.plot(x3, y3, 'bo-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Optimized Path')

    for i, (x, y) in enumerate(zip(x3, y3)):
        plt.text(x, y, str(i), color='red', fontsize=10, ha='center', va='center')

    end_time = time.time()
    execution_time = end_time - start_time
    print("程式執行時間：", execution_time, "秒")

    plt.show()

    print(f"Generation {num_generations}: Best Fitness = {best_fitness}")

if __name__ == "__main__":
    main()
