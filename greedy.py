import random
import time
import math
import matplotlib.pyplot as plt

# 建立起點和終點
start = (0, 0)
end = (100, 100)

# 建立隨機的 10 個中繼站
# intermediates = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]

intermediates = [(0, 0), (56, 42), (49, 91), (27, 50), (2, 45), (33, 41), (34, 67), (34, 91), (31, 94), (56, 77), (66, 77), (100, 100)]

# 計算兩點之間的距離
def calculate_distance(point1, point2):
  return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# 貪心算法來尋找最短路徑
def greedy_shortest_path(start, intermediates, end):
  path = [start]
  while intermediates:
    # 從剩餘的中繼站中找出離當前位置最近的點
    next_point = min(intermediates, key=lambda x: calculate_distance(path[-1], x))
    intermediates.remove(next_point)
    path.append(next_point)
  path.append(end)
  return path



# 計算總路徑的距離
def total_distance(path):
    return sum(calculate_distance(path[i], path[i+1]) for i in range(len(path)-1))

# 繪製路徑圖
def draw_path(path):
  plt.figure(figsize=(10,10))
  plt.scatter(*zip(*path), color='red')  # 將點畫在圖上
  plt.plot(*zip(*path), color='blue')  # 將路徑畫在圖上

  # 標示點的順序
  for i, point in enumerate(path):
      plt.text(point[0], point[1], str(i), fontsize=12, ha='right')
  
  plt.show()

def main():

    start_time = time.time()
    # 使用貪心算法找到最短路徑
    path = greedy_shortest_path(start, intermediates, end)

    # 輸出最短路徑和其距離
    print('最短路徑:', path)
    print('最短路徑的距離:', total_distance(path))

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    # 繪製路徑圖
    draw_path(path)

if __name__ == '__main__':
    main()
