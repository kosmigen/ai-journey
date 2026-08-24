import numpy as np

# 创建一个包含 10 个 0 的一维数组
print(np.zeros(10))

# 创建一个包含 10 个 1 的一维数组
print(np.ones(10))

# 创建一个包含 10 个 10 的一维数组
print(np.full(10, 10))

# 将 python 列表转换为 np 数组
a = np.array([1, 3, 4, 2])

# 访问、修改数组的元素都和使用 py 中的列表一致
print(a[2])

# 创建一个范围值的数组，和 range 使用方法完全一致
print(np.arange(10))
print(np.arange(3, 6))
print(np.arange(1, 9, 4))

# 创建一个从 start 到 stop，一共 num 个元素的数组，会将 [start, stop] 平均分为 num 个元素
print(np.linspace(0, 1, 11))

# 上述的所有例子都是创造的一维数组，我们可以进行创造二维乃至多维数组
# 只用将上述方法中的参数理解为描述数组的形状即可

print(np.zeros([4, 3, 2]))

a = np.array([
    [3, 2, 1],
    [2, 1, 9],
    [7, 0, 4]
])

print(a)
print(a[2][1])

# 表示取第 2 列，第一个参数使用 : 表示取所有行的元素
print(a[:, 1])
print(a[:])

# 创建一个元素随机在 0-1 之间的数据，函数参数描述数组的形状
print(np.random.rand(5))

# 使用表达式乘上 100，表示的是数组中的每个元素都乘上 100
print(np.random.rand(3) * 100)

# 如果需要伪随机数，则需要在生成之前设置随机数种子
# 这样不管生成多少次，生成的随机数都是一样的
np.random.seed(10)
print(np.random.randn(5))

# 随机生成正态分布数
print(np.random.randn(5, 5))

# 随机生成一个 size 的数组，其中元素的最小值和最大值都可以进行设置
print(np.random.randint(1, 5, [5, 6]))

# 如果需要对数组中的每个元素进行操作，直接在外部和数组进行操作即可
a = np.array([
    [1, 2, 3],
    [2, 0, 1],
    [8, 5, 2]
])

print(a)
print(a + 1, a - 1, a * 10, a / 3)

# 判断数组中的元素是否满足条件，会得到一个包含 True/False 元素的数组
print(a > 2)

a = np.arange(1, 5)
b = np.arange(0, 4)

print(a > b)
# 获取符合条件的元素
print(a[a < b])

# mean 得到的是平均值，std 得到的是标准差
print(a.min(), a.max(), a.sum(), a.mean(), a.std())

a = np.array([
    [1, 4, 1],
    [5, 3, 1]
])

# where 可以在满足条件时为 x，不满足条件时为 y，返回得到的新数组
a = np.where(a > 3, 10, 0)

print(a)