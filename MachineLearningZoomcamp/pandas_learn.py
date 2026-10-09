import pandas as pd

# pandas 用于数据分析和操作

data = [
    ["刘杰", 22, "男", "回龙乡牛家村"],
    ["龚媛", 22, "女", "丹山镇"]
]

col = ["姓名", "年龄", "性别", "居住地址"]

# 给定列名和对应的数据可以得到一个表格
print(pd.DataFrame(col, data))

data = [
    {
        "姓名": "刘杰",
        "年龄": 22,
        "储蓄": 3120,
        "性别": "男",
        "居住地址": "回龙乡牛家村"
    },
    {
        "姓名": "龚媛",
        "年龄": 22,
        "储蓄": 2520,
        "性别": "女",
        "居住地址": "丹山镇"
    }
]

# 可以从字典中推断出来列表中名称是什么，有什么值
df = pd.DataFrame(data)

print(df)

# head 方法默认返回前 5 条数据，也可以进行指定返回的数据条数
print(df.head())

# 可以通过 .列名 访问对应的数据，意思是得到的 df 是表格数据的抽象对象
# 输出中的两个部分是相同的
print(df.姓名, df["姓名"])

# 可以使用一个二维数组选定我们需要的列数据
print(df[["性别", "居住地址"]])

# 可以在 df 中进行添加列
df["id"] = [1, 2]
print(f"{"-" * 20}")
print(df)

# 删除一行
del df['id']

# 访问 df 的 index 属性
# index 是表示多少条数据的标志（索引）
print(f"{"-" * 20}")
print(df.index)
# 我们甚至可以修改这个 index 的表达
df.index = ['a', 'b']
print(f"{"-" * 20}")
print(df)

df.index = [0, 1]

# 使用 df.loc[index] 可以获取对应行索引的数据
# 也可以使用列表进行说明获取数据的行索引范围
print(f"{"-" * 20}")
print(df.loc[1])
print(f"{"-" * 20}")
print(df.loc[[0, 1]])

# 当 index 变为不是 0，1，2，... 时，为了不变化后续的 loc 中指定的数据索引，可以使用 iloc，这个方法始终用的是 0，1，2，...

# 重置索引为 0，1，2，...，并保留之前改为的索引值，列名为 index，使用 reset_index 方法
# 这个方法是返回一个新的 df 对象
# 如果不想要 index 列，则可以设置参数 drop 为 True
# df = df.reset_index()
df = df.reset_index(drop=True)
print(f"{"-" * 20}")
print(df)

# 可以对某一列进行操作，和 numpy 类似
print(f"{"-" * 20}")
df['年龄'] *= 2
print(df['年龄'])

# 可以使用比较运算符，得到的是一个布尔类型的返回值
# 可以通过这个得到满足某个条件的结果
print(df['年龄'] > 30)
print(df[df['年龄'] > 40])

# 对字符串进行操作
df['英语姓名'] = ['Tom Jay', 'Jney May']
print(f"{"-" * 20}")
print(df)
print(f"{"-" * 20}")
# 会返回一个修改后的 Series 对象
print(df['英语姓名'].str.lower())
# 虽然进行修改，但是得到的是一个新的对象，所以我们还需要将更新的对象写回 df 对象中
df['英语姓名'] = df['英语姓名'].str.replace(' ', '_').str.upper()
print(df)

print(f"{"-" * 20}")
print(df['年龄'].describe())

# 调用一列中的不同种类的方法 nunique()
print(f"{"-" * 20}")
print(df['年龄'].nunique())

# 调用 df 的 isnull 方法，可以得到一个布尔的表格，表达其是否为空
print(f"{"-" * 20}")
print(df.isnull())

# 可以通过 groupby 对数据指定的列进行分组，然后对每个分组后的数据进行一系列的操作
print(df.groupby('年龄')["储蓄"].max())

# 获取底层的 numpy 数组，通过使用 values
print(type(df['年龄'].values))

df.to_records()

df.to_dict()