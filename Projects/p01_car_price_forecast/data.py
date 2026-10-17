from typing import NamedTuple
import pandas as pd
import numpy as np
from pandas import DataFrame

"""
获取处理好的数据
"""

class DivideResult(NamedTuple):
    train_data: DataFrame
    validation_data: DataFrame
    test_data: DataFrame

def get_car_data() -> DivideResult:
    """
    将汽车数据集划分为 train validation test 三个部分
    :return: 三个部分集合组成的元组
    """
    data = pd.read_csv('data.csv', encoding='utf-8')

    # 统一数据的列名表达
    data.columns = data.columns.str.lower().str.replace(" ", "_")
    # 打乱排序
    ids = np.arange(len(data))
    np.random.seed(10)
    np.random.shuffle(ids)
    # 根据打乱的数组分配数据
    return _assign_data(data, ids)

def _assign_data(data, shuffle_array, validation_weight = 0.2, test_weight = 0.2) -> DivideResult:
    """
    根据 shuffle_array 打乱 data，然后根据 weight 得到不同数据集
    重置数据集中的 id
    :param data: 整体数据集
    :param shuffle_array: 打乱后的 id 数组
    :param train_weight:
    :param validation_weight:
    :param test_weight:
    :return:
    """
    data_len = len(data)
    val_len = int(data_len * validation_weight)
    test_len = int(data_len * test_weight)
    validation_data = data.iloc[shuffle_array[: val_len]]
    test_data = data.iloc[shuffle_array[val_len : val_len + test_len]]
    train_data = data.iloc[shuffle_array[val_len + test_len :]]

    validation_data = validation_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    return DivideResult(train_data, validation_data, test_data)

if __name__ == "__main__":
    print(get_car_data())