from pandas import Series
from typing import Tuple, Dict, Union
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from .distribution_tools import outstanding_1_check, outstanding_2_check

def single_point_schema_check(data: Series) -> bool:
    """
    This function is used to check the schema of the data.

    Args:
        data (Series): The data to be checked.
            - The data should be a pandas Series with float64 or int data type.

    Returns:
        bool: A boolean value indicating whether the schema of the data is correct.
            - True: The schema of the data is correct.
            - False: The schema of the data is incorrect.
    """
    try:
        if data.dtype != 'float64' or data.dtype != 'int': # float64 || int
            data = data.astype(float)
    except Exception :
        return False

    return True


def attribution_detection(data: Series) -> Tuple[Dict[str, Union[int, float]], str]:
    """
    This function takes in a Series and returns a dictionary and a string of the attribution element of the data.

    Args:
        data (Series): The data to be analyzed.

    Returns:
        Dict[str, Union[int, float]]:
            - The dictionary contains the "index" and "ratio" of the attribution element.
            - The "index" is the index of the attribution element.
            - The "ratio" is the ratio of the attribution element to the total data.
            - If the attribution element is not found, the dictionary is {"index": -1}.
        str:
            - The string contains the explanation of the attribution element.
            - If the attribution element is not found, the string is "None".
    """
    count_negative = (data < 0).sum()
    if count_negative !=0:
        data = data[data >= 0]
    if len(data) == 1:
        return {"index": data.index[0]}, \
                f"仅有一个正值，需要进一步分析"

    ratio = round(data.max() * 100 / data.sum() ,2)
    id_max = data.idxmax()
    if ratio > 50 and ratio < 60:
        return {"index": id_max, "ratio": ratio}, \
                f"贡献量超过总量的一半（{ratio}%）"
    elif ratio > 60:
        num = int(ratio // 10)
        to_word = {
            6: "六",
            7: "七",
            8: "八",
            9: "九",
            10: "十"
        }
        return {"index": id_max, "ratio": ratio}, \
                f"贡献量达到{to_word.get(num, '半')}成（{ratio}%）"
    elif data.max()*2 == data.sum():
        return {"index": id_max, "ratio": 50}, \
                f"贡献量正好达到总量的一半"
    else:
        return {"index": -1}, \
                None


def outstanding_1_detection(data: Series, threshold: float = 0.1, beita: float = 0.7) -> Tuple[Dict[str, Union[int, float]], str]:
    """
    This function takes in a Series and returns a dictionary and a string of the outstanding_1 element of the data.

    Args:
        data (Series): The data to be analyzed.

    Returns:
        Dict[str, Union[int, float]]:
            - The dictionary contains the "index", "ratio_to_second" and "ratio_to_min" of the outstanding_1 element.
            - The "index" is the index of the outstanding_1 element.
            - The "ratio_to_second" is the ratio of the outstanding_1 element to the second highest value.
            - The "ratio_to_min" is the ratio of the outstanding_1 element to the lowest value.
            - If the outstanding_1 element is not found, the dictionary is {"index": -1}.
        
        str:
            - The string contains the explanation of the outstanding_1 element.
            - If the outstanding_1 element is not found, the string is "None".
    """

    if len(data) <= 1:
        return {"index": -1}, \
                None

    count_negative = (data < 0).sum()
    if count_negative !=0:
        data = data[data >= 0]

    mean = data.mean()
    count = (data > mean).sum()


    if count == 1 and outstanding_1_check(data=data, threshold=threshold, beita=beita) == True:
        id_max = data.idxmax()
        value_max = data.max()
        ratio_to_second = int(value_max // data[data != value_max ].max())
        ratio_to_min = int(value_max // data.min())
        # if ratio_to_min == ratio_to_second:
        if ratio_to_min == ratio_to_second:
            return {"index": id_max, "ratio_to_second": ratio_to_second, "ratio_to_min":ratio_to_min }, \
                    f"是其余数据的{ratio_to_second}倍多"
        else:
            return {"index": id_max, "ratio_to_second": ratio_to_second, "ratio_to_min":ratio_to_min }, \
                    f"是其余数据的{ratio_to_second}到{ratio_to_min}倍"
    else:
        return {"index": -1}, \
                None


def outstanding_2_detection(data: Series, threshold: float = 0.3, beita: float = 0.7) -> Tuple[Dict[str, Union[int, float]], str]:
    """
    This function takes in a Series and returns a dictionary and a string of the outstanding_2 element of the data.

    Args:
        data (Series): The data to be analyzed.

    Returns:
        Dict[str, Union[int, float]]:
            - The dictionary contains the "index", "index_second" and "ratio" of the outstanding_2 element.
            - The "index" is the index of the outstanding_2 element.
            - The "index_second" is the index of the second highest value.
            - The "ratio" is the ratio of the outstanding_2 element to the total data.
            - If the outstanding_2 element is not found, the dictionary is {"index": -1}.
        str:
            - The string contains the explanation of the outstanding_2 element.
            - If the outstanding_2 element is not found, the string is "None".
    """
    if len(data) <= 2:
        return {"index": -1}, \
                None

    count_negative = (data < 0).sum()
    if count_negative !=0:
        data = data[data >= 0]

    mean = data.mean()
    count = (data > mean).sum()

    if count == 2 and outstanding_2_check(data=data, threshold=threshold, beita=beita) == True:
        top_two_values_data = data.nlargest(2)
        ratio = round( top_two_values_data.sum() * 100 / data.sum() ,2)
        return {"index": data.idxmax(), "index_second": top_two_values_data.index[1], "ratio": ratio}, \
                f"占总体的绝大部分（{ratio}%）"
    else:
        return {"index": -1}, \
                None


def outstanding_last_detection(data: Series, threshold: float = 0.3, beita: float = 0.7) -> Tuple[Dict[str, Union[int, float]], str]:

    if len(data) <= 1:
        return {"index": -1}, \
                None

    negative_data = data[data < 0].sort_values()
    count_negative = len(negative_data)

    if count_negative == 0:
        return {"index": -1}, \
                None
    elif count_negative == 1:
        return {"index": negative_data.index[0]}, \
                f"仅有一个负值，需要进一步分析"
    else:
        positive_data = -negative_data
        return outstanding_1_detection(positive_data, threshold=threshold, beita=beita)


def evenness_detection(data: Series) -> Tuple[Dict[str, Union[int, float]], str]:
    """
    This function takes in a Series and returns a dictionary and a string of the evenness information of the data.

    Args:
        data (DataFrame): 
            - The data to be analyzed.

    Returns:
        Dict[str, Union[int, float]]:
            - The dictionary contains the "index" and "mean" of the evenness information.
            - The "index" is the index of the maximum value.
            - The "mean" is the mean of the data.
            - If data is not evenly distributed, the dictionary is {"index": -1}.
        str:
            - The string contains the explanation of the evenness information.
            - If the evenness information is not found, the string is "None".
    """
    
    # 计算偏度:::出现精度消失的问题
    # skewness = skew(data_series)
    
    # 计算变异系数（CV，即相对标准差）
    # mean_value = data_series.mean()
    # std_dev = data_series.std(ddof=1)  # 使用无偏估计计算标准差
    # cv = std_dev / mean_value
    
    # print(data_series.mean())

    if len(data) <= 2:

        return {"index": -1}, \
                None

    count_negative = (data < 0).sum()
    count_positive = (data > 0).sum()
    if count_negative !=0 and count_positive !=0:
        return {"index": -1}, \
                None
    

    diff_max_mean = (data.max() - data.mean()) * 100.0 / data.sum()
    diff_min_mean = (data.mean() - data.min()) * 100.0 / data.sum()
    
    # print("偏度:", skewness)
    # print("变异系数:", cv)
    # print("最大值和最小值的差值占平均值的百分比:", diff_max_mean, diff_min_mean)

    # 判断数据是否平均
    is_average = (
        # abs(skewness) <= 0.5 and  # 偏度阈值
        # cv <= 0.5 # 变异系数阈值
        # and
        diff_max_mean <= 3    # 最大值最小值在平均值上下3%以内
        and
        diff_min_mean <= 3
    )
    # print("是否平均:", is_average)
    if is_average and outstanding_1_check(data) == False and outstanding_2_check(data) == False:
        return {"index": data.idxmax(), "mean": data.mean()}, \
                f"分布平均"
    else:
        return {"index": -1, }, \
                None