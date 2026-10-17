import numpy as np
from typing import NamedTuple
from data import get_car_data, DivideResult
import pandas as pd

class YData(NamedTuple):
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    data: DivideResult

def _pre_delete_msrp() -> YData:
    """
    删除汽车数据中的建议零售价
    :return: 不同汽车集合建议零售价取对数组合成的元组和处理后的数据
    """
    divide_data = get_car_data()
    y_train = np.log1p(divide_data.train_data.msrp.values)
    y_validation = np.log1p(divide_data.validation_data.msrp.values)
    y_test = np.log1p(divide_data.test_data.msrp.values)
    print(y_train, y_validation, y_test, sep='\n')

    del divide_data.train_data['msrp']
    del divide_data.validation_data['msrp']
    del divide_data.test_data['msrp']

    return YData(y_train, y_validation, y_test, divide_data)

def _dot(xi, w):
    """
    两个向量的点乘
    :param xi: 特征值
    :param w: 权重
    :return:
    """
    x_len = len(xi)
    w_len = len(w)
    if w_len != x_len:
        return None

    result = 0
    for i in range(len(xi)):
        result += xi[i] * w[i]

    return result

# 如果将每辆汽车的特征值抽取，得到一列的数据，则可以形成 X = [x1,x2,x3,...] 的矩阵
# 对于每个特征值的权重有 [w] （上述提到的向量都是列向量）
# 为了让每辆车都可以和权重想乘，则有 X 的转置 . w = y

# Xw = y -> X^{T}Xw = X^{T}y -> w = (X^{T}X)^{-1}X^{T}y
# 若 X 为 [8,5]，则 XX^{T} 就为 [8,8]，
# 所以 w 的求解就结束了
def train_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X 必须是二维数组")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X 和 y 的样本数不一致")

    ones = np.ones((X.shape[0], 1))
    X_aug = np.column_stack((ones, X))

    XTX = X_aug.T.dot(X_aug)
    w = np.linalg.solve(XTX, X_aug.T.dot(y))
    return w


def test_linear_regression():
    X = [
        [123, 98, 54, 129, 90],
        [210, 131, 34, 123, 140],
        [323, 81, 14, 240, 190],
        [76, 112, 44, 148, 187],
        [183, 98, 54, 129, 90],
        [210, 131, 34, 123, 140],
        [133, 91, 14, 240, 190],
        [200, 111, 44, 168, 187],
    ]

    X = np.array(X)

    # dot 进行矩阵的乘积，使用 narray.T 获得矩阵的对应转置
    XXT = X.T.dot(X)

    # 得到 XXT 的逆矩阵
    XXT_inv = np.linalg.inv(XXT)

    print(XXT_inv.dot(XXT).round(1))

    y = np.array([3.2, 2.1, 4, 3.9, 2.9, 3.8, 4, 3])
    print(f"训练得到 w 为：{XXT_inv.dot(X.T).dot(y)}")
    ones = np.ones(X.shape[0])
    X = np.column_stack((ones, X))
    XXT = X.T.dot(X)

    # 得到 XXT 的逆矩阵
    XXT_inv = np.linalg.inv(XXT)

    y = np.array([3.2, 2.1, 4, 3.9, 2.9, 3.8, 4, 3])
    w = XXT_inv.dot(X.T).dot(y)

    print(f"偏置值为：{w[0]}")
    print(f"训练得到 w 为：{w[1:]}")

def baseline_model(deal_data: YData):
    base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
    df = deal_data.data.train_data

    # 注意判断是否存在 nan，如果计算的数值中存在 nan，则结果必然为 nan
    # 目前是将 nan 全部替换为 0，但是替换为 0 真的好吗？xi 为 0，则这个 wi 会收到影响，很难训练到这个权重
    X_train = df[base].fillna(0)
    y_train = deal_data.y_train
    return train_linear_regression(X_train, y_train), X_train

if __name__ == "__main__":
    deal_data = _pre_delete_msrp()
    # print(_dot([1, 2, 3], [0.1, 0.2, 0.3]))
    # test_linear_regression()
    w_full, X_train = baseline_model(deal_data)
    print(w_full[0], w_full[1:])

    y_pred = w_full[0] + X_train.dot(w_full[1:])
    # print(f"根据计算后的权重得到的预测值 {y_pred}")
    # print(f"真实值 {deal_data.y_train}")
    print(f"真实值与预测值的差值：{deal_data.y_train - y_pred}，差值平均值：{(deal_data.y_train - y_pred).mean()}")