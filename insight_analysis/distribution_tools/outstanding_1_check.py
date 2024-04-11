from pandas import Series
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

def outstanding_1_significance(data: Series, beita=0.7) -> float:
    """
    Calculate the probability of the given data to be an outstanding 1-significance distribution.

    Args:
        data (Series): The data to be analyzed.

    Returns:
        float: The significance of the max_value to be outstanding_1.
    """

    numpy_array = data.values
    Y = np.sort(numpy_array)[::-1]

    i = np.arange(1, len(Y) + 1)
    X = i ** (-beita)

    model = LinearRegression(fit_intercept=False)
    model.fit(X.reshape(-1, 1), Y)
    Y_pred = model.predict(X.reshape(-1, 1))
    # print(Y_pred)

    # 计算残差
    residuals = Y - Y_pred
    # print("residuals\n", residuals)

    # 计算残差的高斯分布参数：均值和标准差
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    # print("mu, sigma", mu, sigma)

    # 计算最大值的残差 R_MAX
    R_MAX = Y[0] - Y_pred[0]

    # 用哪一个？概率论学完就忘
    # 计算 R_MAX 对应的概率密度值
    # p_value = stats.norm.pdf(R_MAX, mu, sigma)

    # 在高斯分布假设下观察到 残差大于 R_MAX 的概率
    p_value = 1 - stats.norm.cdf(R_MAX, mu, sigma)

    # print("p_value: ", p_value)

    return p_value

def outstanding_1_check(data: Series, threshold=0.3, beita=0.7) -> bool:
    """
    Check if the given data is an outstanding 1-significance distribution.

    Args:
        data (Series): The data to be analyzed.
        threshold (float, optional): The threshold of the significance. Defaults to 0.3.

    Returns:
        bool: True if the data is an outstanding 1-significance distribution, False otherwise.
    """

    p_value = outstanding_1_significance(data, beita=beita)
    if __name__ == '__main__':
        print("p_value: ", p_value, end="\t")

    if p_value < threshold:
        return True
    else:
        return False
    

if __name__ == '__main__':
    print(outstanding_1_check(Series([1, 2, 3, 4, 5, 6]), threshold=0.3))  
    # p_value:  0.8053498037781665    False
    print(outstanding_1_check(Series([1, 2, 3, 4, 10, 20]), threshold=0.3))
    # p_value:  0.4406144626376285    False
    print(outstanding_1_check(Series([1, 2, 3, 4, 5, 20]), threshold=0.3))
    # p_value:  0.16749167805184806   True
    print(outstanding_1_check(Series([2, 4, 6, 8, 500]), threshold=1/5, beita=0.5))
    # p_value:  0.1876366388603916    True