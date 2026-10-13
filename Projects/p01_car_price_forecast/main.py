import numpy as np
import pandas as pd

# 可以使用 seaborn 库进行数据的可视化

def preprocessing(car_data):
    print(car_data)

    # 处理数据的列名，进行统一
    car_data.columns = car_data.columns.str.lower().str.replace(" ", "_")
    print(car_data)

    # 获取每列的前 5 个元素
    for i in car_data:
        print(i)
        # 返回前几个元素，其实没有太多的价值，可以使用 unique 得到前几个不同的数据
        print(car_data[i].head())
        print(f"{'-' * 5} unique {'-' * 5}")
        print(car_data[i].unique()[:5])
        print(f"一共有 {car_data[i].nunique()} 个不同数据")
        print()

    # 在每列中的不同数据之间，有时相差非常大，这种差距会给机器学习带来不好的影响
    # 所以，使用 np 中的对数函数将数据之间的差距缩小
    print(car_data.msrp)
    print(np.log1p(car_data.msrp))

    # 查看缺失值的个数
    print(car_data.isnull().sum())


if __name__ == "__main__":
    car_data = pd.read_csv('data.csv')
    preprocessing(car_data)