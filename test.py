import math

# p_list = [0.01, 0.15, 0.12, 0.03, 0.02, 0.04, 0.02, 0.04, 0.01, 0.13, 0.15, 0.14, 0.11, 0.03]
p_list = [0.25, 0.2, 0.15, 0.1, 0.08, 0.08, 0.05, 0.04, 0.03, 0.02]
# 对列表进行排序
sorted_p_list = sorted(p_list)

print(sorted_p_list)
# 计算信息熵
entropy = 0
for p in p_list:
    entropy -= p * math.log2(p)

print(entropy)
