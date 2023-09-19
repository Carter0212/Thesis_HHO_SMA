N = 100  # 維度數量
lower_bound = 0  # 下界
upper_bound = 1000  # 上界

# 將上界轉換為二進制
upper_bound_binary = bin(upper_bound)[2:].zfill(N)

# 初始化維度組合的列表
combinations = []

# 使用二進制表示生成所有可能的維度組合
for i in range(2 ** N):
    binary_representation = bin(i)[2:].zfill(N)
    combination = [int(binary_digit) * (upper_bound - lower_bound) + lower_bound for binary_digit in binary_representation]
    combinations.append(combination)

# 在這裡執行你的適應函數或其他操作
# 例如，計算適應函數值
def fitness_function(combination):
    ans=1
    for i in combination:
        ans*=abs(i)
    return sum(abs(num) for num in combination) + ans
    # 在這裡計算適應函數值，combination是一個包含N個維度值的列表
    # 你可以使用for迴圈來計算總值、總權重等

# 例如，計算適應函數的值
fitness_values = [fitness_function(combination) for combination in combinations]

# 找到最佳結果
best_index = fitness_values.index(max(fitness_values))
best_combination = combinations[best_index]

print("Best combination (binary representation):", "".join(str(int(x >= (upper_bound - lower_bound) // 2)) for x in best_combination))
print("Best fitness value:", fitness_values[best_index])
