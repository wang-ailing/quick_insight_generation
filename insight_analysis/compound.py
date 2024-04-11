import numpy as np
from scipy import stats


def correlation_detection(X, Y, threshold=0.05):

    # 计算Pearson相关系数
    r, p_value = stats.pearsonr(X, Y)

    if __name__ == '__main__':
        # 输出相关系数和p值
        print(f"Pearson Correlation Coefficient: {r}")
        print(f"P-value: {p_value}")

    # 判断相关性的显著性
    if p_value < threshold:
        print("The correlation is statistically significant at the 0.05 level.")
    else:
        print("There is not enough evidence to suggest a statistically significant correlation.")


if __name__ == '__main__':
    # same change
    X = np.array([1, 2, 7, 8, 1])
    Y = np.array([5, 6, 15, 18, 6])
    correlation_detection(X, Y, threshold=0.05)