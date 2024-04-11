from pandas import Series
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from typing import Tuple

def outstanding_2_significance(data: Series, beita:float=0.7) -> float:
    """
    Calculate the probability of the given data to be an outstanding 2-significance distribution.

    Args:
        data (Series): The data to be analyzed.

    Returns:
        float: The significance of the max_1_value and the max_2_value to be outstanding_2.
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

    # 计算最大值和第二大值的残差和 R_MAX
    R = (( Y[0] - Y_pred[0] ) + ( Y[1] - Y_pred[1] ) ) / 2

    # 计算 R_MAX 对应的概率密度值 -> Nope! 不应该这样算
    # p_value = stats.norm.pdf(R, mu, sigma)

    # 在高斯分布假设下观察到 残差大于 R 的概率   越小越显著
    p_value = 1 - stats.norm.cdf(R, mu, sigma)

    # print("p_value: ", p_value)

    # oustanding_2_plot(Y, a=model.coef_[0], b = -0.7)
    # R_MAX = ( Y[0] - Y_pred[0] ) 
    # R_MIN = ( Y[1] - Y_pred[1] ) 
    # print("R_MAX: ", R_MAX, "R_MIN: ", R_MIN)
    # outstanding_2_Gaussian_plot(mu, sigma, (Y[0]-Y_pred[0]), (Y[1]-Y_pred[1]), R)

    return p_value

def outstanding_2_check(data: Series, threshold:float=0.3, beita:float=0.7) -> bool:
    """
    Check if the given data is an outstanding 1-significance distribution.

    Args:
        data (Series): The data to be analyzed.
        threshold (float, optional): The threshold of the significance. Defaults to 0.3.

    Returns:
        bool: True if the data is an outstanding 1-significance distribution, False otherwise.
    """

    p_value = outstanding_2_significance(data, beita=beita)
    # print("p_value: ", p_value, end="\t")

    if p_value < threshold:
        return True
    else:
        return False

def outstanding_2_Gaussian_plot(mu, sigma, r1, r2, R) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成一系列的x值
    x = np.linspace(mu-3*sigma, mu+3*sigma, 1000)

    # 计算对应的高斯函数值
    y = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))

    # 选择一个特定的x值
    # x_points = [r1, r2] 

    plt.plot(x, y)
    plt.title('Gaussian Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # y_point = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x_points[0]-mu)**2)/(2*sigma**2))
    # plt.plot(x_points[0], y_point, 'ro', )  # 'ro'表示红色圆点
    # plt.annotate('$r1 = {:.2f}$'.format(x_points[0]),
    #             (x_points[0], y_point),  # 标签的位置
    #             textcoords="offset points",  # 相对位置
    #             xytext=(0,10),  # 文本标签的偏移量
    #             ha='center')  # 水平居中对齐
    # y_point = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x_points[1]-mu)**2)/(2*sigma**2))
    # plt.plot(x_points[1], y_point, 'ro', )  # 'ro'表示红色圆点
    # plt.annotate('$r2 = {:.2f}$'.format(x_points[1]),
    #             (x_points[1], y_point),  # 标签的位置
    #             textcoords="offset points",  # 相对位置
    #             xytext=(0,10),  # 文本标签的偏移量
    #             ha='center')  # 水平居中对齐

    y_point = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((R-mu)**2)/(2*sigma**2))
    plt.plot(R, y_point, 'ro', )  # 'ro'表示红色圆点
    plt.annotate('$r1 = {:.2f}$'.format(R),
                (R, y_point),  # 标签的位置
                textcoords="offset points",  # 相对位置
                xytext=(0,10),  # 文本标签的偏移量
                ha='center')  # 水平居中对齐
    # 绘制高斯函数图像

    # plt.axhline(0, color='black',linewidth=0.5)
    # plt.axvline(0, color='black',linewidth=0.5)
    x_point = R
    x_values = x
    y_values = y
    y_values_after = np.where(x_values > x_point, y_values, np.zeros_like(x_values))
    y_values_zero = np.zeros_like(x_values)

    # 使用fill_between函数填充阴影
    plt.fill_between(x_values, y_values_zero, y_values_after, where=(x_values > x_point), color='red', alpha=0.7, label='p_value < 0.1')
    plt.ylim(ymin=0)

    # 添加图例
    plt.legend()

    plt.axvline(x=mu, color='r', linestyle='--')
    plt.savefig("outstanding_2_Gaussian_plot.png",    dpi=800,    bbox_inches='tight')
    plt.show()

def oustanding_2_plot(values: list, a: float, b: float) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # 示例数据
    # sorted_x_ray = np.arange(1, len(values)+1)

    start_value = 1
    end_value = len(values)+0.5
    step_size = 0.01

    x = np.arange(start_value, end_value, step_size)

    # 计算Y轴数据，即幂律函数的值
    y = a * x**b

    # 创建一个新的图表
    plt.figure()

    # 绘制条形图
    plt.bar(np.arange(1, len(values)+1), values, color='skyblue')

    # 绘制幂律函数的曲线
    # plt.plot(x, y, 'blue', linewidth=1)
    plt.plot(x, y, "blue", linewidth=1, label=f"y = {a:.2f} * x^({b:.2f})")
    plt.legend()

    # 添加标题和标签
    plt.title('Bar Chart with Power Law Curve')
    plt.xlabel('Sorted Values')
    plt.ylabel('Values')

    # 可以添加网格线以便更好地观察曲线与条形的关系
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("outstanding_2_plot.png",    dpi=800,    bbox_inches='tight')
    # 显示图表
    plt.show()
    

if __name__ == '__main__':

    # print(outstanding_2_check(Series([1, 2, 3, 4, 5, 6]), threshold=0.1))
    # p_value:  0.5023839946579908    False
    # print(outstanding_2_check(Series([1, 2, 3, 4, 10, 20]), threshold=0.1))
    # # p_value:  0.0981688097939577    True
    print(outstanding_2_check(Series([1, 2, 3, 4, 15, 20]), threshold=0.1))
    # # p_value:  0.08078214808217588   True
    # print(outstanding_2_check(Series([1, 2, 3, 4, 5, 20]), threshold=0.1))
    # # p_value:  0.22370415276637567   Fase