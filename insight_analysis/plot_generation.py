import numpy as np
from scipy import interpolate
from scipy.interpolate import Akima1DInterpolator
import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_generation_Akima(x, y):
    f = Akima1DInterpolator(x, y)
    x_new = np.linspace(x.min(), x.max(), 100)
    y_new = f(x_new)

    # # 示例数据点
    # x = np.array([1, 2, 3, 4, 5])
    # y = np.array([2, 4, 10, 8, 10])

    # # 创建插值函数
    # f = interpolate.interp1d(x, y, kind='zero')

    # # 生成新的数据点
    # x_new = np.linspace(min(x), max(x), 100)

    # # 使用插值函数计算新数据点的值
    # y_new = f(x_new)

    # 绘制原始数据点和插值曲线
    plt.plot(x, y, 'o', label='Original data')
    plt.plot(x_new, y_new, label='Curve Interpolation')
    plt.legend()
    # plt.title('Cubic Interpolation with scipy.interpolate.interp1d')
    if __name__ == '__main__':
        plt.savefig('../pic/曲线插值.png', dpi=800)
        
    plt.show()

if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 10, 8, 10])
    plot_generation_Akima(x, y)