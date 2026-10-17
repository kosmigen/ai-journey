import numpy as np
from typing import NamedTuple
from data import get_car_data

class YData(NamedTuple):
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray

def _pre_delete_msrp() -> YData:
    """
    删除汽车数据中的建议零售价
    :return: 不同汽车集合建议零售价取对数组合成的元组
    """
    divide_data = get_car_data()
    y_train = np.log1p(divide_data.train_data.msrp.values)
    y_validation = np.log1p(divide_data.validation_data.msrp.values)
    y_test = np.log1p(divide_data.test_data.msrp.values)
    print(y_train, y_validation, y_test, sep='\n')

    del divide_data.train_data['msrp']
    del divide_data.validation_data['msrp']
    del divide_data.test_data['msrp']

    return YData(y_train, y_validation, y_test)

if __name__ == "__main__":
    _pre_delete_msrp()