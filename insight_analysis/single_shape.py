import pandas as pd
from pandas import DataFrame
from typing import Tuple, List, Union, Dict
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import ruptures as rpt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from alibi_detect.od import SpectralResidual
import pymannkendall as mk

def single_shape_schema_check(data: DataFrame) -> bool:
    """
    Check if the data has the correct schema for a single shape.

    Args:
        data (DataFrame): DataFrame with columns 'date' and 'value'

    Returns:
        bool: True if the data has the correct schema, False otherwise.
    """

    try:
        data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d') 
        # date example: '2021-01-01'
        if data.iloc[:, 1].dtype != 'float64':
            data.iloc[:, 1] = data.iloc[:, 1].astype(float)
    except Exception :
        return False

    return True


def change_point_detection(data: DataFrame, model: str = 'rbf', jump: int = 1, penalty: float = 1.0) -> List[int]:
    values = data.iloc[:, 1].values
    # change point detection
    model = "l2"  # "l1", "l2", "rbf"
    algo = rpt.Pelt(model=model, min_size=7, jump=1).fit(values)
    my_bkps = algo.predict(pen=5)
    # show results
    if __name__ == '__main__':
        print(my_bkps)
    # result = my_bkps[:-1]
    result = my_bkps
    # 显示结果
    # plt.plot(values)
    # for bkp in result:
    #     plt.axvline(x=bkp, color='r', linestyle='--')
    # plt.show()
    return result

def outlier_detection(data: DataFrame, image_path: str = None) -> List[bool]:
    """
    Detect outliers using Spectral Residual algorithm.
    Please refer to https://docs.seldon.io/projects/alibi-detect/en/latest/api/alibi_detect.od.html#alibi_detect.od.SpectralResidual for   for more details.
    PaperTitle: Time-Series Anomaly Detection Service at Microsoft 
    URL: https://arxiv.org/abs/1906.03821

    Args:
        data (DataFrame): DataFrame with columns 'date' and 'value'

    Returns:
        List[bool]: List of boolean values indicating whether each data point is an outlier or not.
    """

    values = data.iloc[:, 1].values

    outlier_detector = SpectralResidual(
        # threshold=None,                  # threshold for outlier score
        threshold=0.99,                  # threshold for outlier score
        window_amp=len (values)//2,                   # window for the average log amplitude
        # window_amp=min (7, len (values)),                   # window for the average log amplitude
        # window_local=min (7, len (values)//2),                 # window for the average saliency map
        window_local=len (values)//3,                 # window for the average saliency map
        n_est_points=len (values)//3,                 # nb of estimated points padded to the end of the sequence
        padding_amp_method='reflect',    # padding method to be used prior to each convolution over log amplitude.
        padding_local_method='reflect',  # padding method to be used prior to each convolution over saliency map.
        padding_amp_side='bilateral'     # whether to pad the amplitudes on both sides or only on one side.
        # padding_amp_side='right'     # whether to pad the amplitudes on both sides or only on one side.
    )

    # outlier_detector.infer_threshold(data['value'].values, threshold_perc=95)
    # if __name__ == '__main__':
    #     print('New threshold: {:.4f}'.format(outlier_detector.threshold))

    preds = outlier_detector.predict(values)

    if __name__ == '__main__':
        # print(preds['data']['is_outlier'])
        plt.plot(data['value'])
        for i in range(len(preds['data']['is_outlier'])):
            if preds['data']['is_outlier'][i]:
                plt.scatter(x=i, y=data['value'][i], color='red', marker='o')
    
        plt.show()

    if image_path is not None:
        plt.plot(data['value'])
        for i in range(len(preds['data']['is_outlier'])):
            if preds['data']['is_outlier'][i]:
                plt.scatter(x=i, y=data['value'][i], color='red', marker='o')
        
        plt.savefig(image_path, bbox_inches='tight', dpi=800)
        # plt.show()

    if len(preds['data']['is_outlier']) > 0:
        return preds['data']['is_outlier']
    else:
        return None


def seasonality_detection(data: DataFrame, period: int = 7, threshold: float = 0.05) -> Tuple[Dict, str]:
    values = data.iloc[:, 1].values

    # Attention! This may not be useful for some cases
    if period == -1: # loop through all possible periods 
        return seasonality_detection_all_period(data, threshold)

    # certain period seasonality detection
    decomposition = seasonal_decompose(values, model='additive', period=period)
    season = decomposition.seasonal

    Y = values[period//2-1:-period//2]
    season = season[period//2-1:-period//2]
    X = sm.add_constant(season)
    Y = np.nan_to_num(Y, nan=0)
    X = np.nan_to_num(X, nan=0)
    model = sm.OLS(Y, X)
    results = model.fit()
    print(results.params)


    # print model summary: a report 
    # print(model.summary())

    # if p-value is less than threshold(0.05 default), then seasonality is significant
    if model.pvalues[1] < threshold:
        return {'period': period,}, f"周期为{period}天的规律表现显著。"
    else:
        return {}, None

def seasonality_detection_all_period(data: DataFrame, threshold: float = 0.05) -> Tuple[Dict, str]:
    values = data.iloc[:, 1].values
    final_period = -1
    compare_p = threshold
    for i in range(3, len(values)//3+1):

        decomposition = seasonal_decompose(values, model='additive', period=i)
        # trend = decomposition.trend
        season = decomposition.seasonal
        # residual = decomposition.resid

        if (i//2-1) < 0 or (i//2-1) > len(values):
            continue
        Y = values[i//2-1:-(i//2)]
        # trend = trend[period//2-1:-period//2]
        season = season[i//2-1:-(i//2)]
        # residual = residual[period//2-1:-period//2]
        # print(Y)
        # print(season)
        # X = sm.add_constant(season)
        X = season
        Y = np.nan_to_num(Y, nan=0)
        X = np.nan_to_num(X, nan=0)
        # print(X)
        # build linear regression model
        model = sm.OLS(Y, X).fit()

        # print model summary: a report 
        # print(model.summary())

        # if p-value is less than threshold(0.05 default), then seasonality is significant
        if model.pvalues[0] < threshold:
            if __name__ == '__main__':

                print(f"周期为{i}天的季节性影响显著。",model.pvalues[0])
            if compare_p > model.pvalues[0]:
                final_period = i
                compare_p = model.pvalues[0]

        else:
            continue
    if final_period!= -1:
        return {'period': i}, f"周期为{i}天的季节性表现最显著。"
    else:
        return {}, None

def trend_detection(data: DataFrame, min_interval: int = 7, alpha: float = 0.05) -> Tuple[Dict, str]:

    values = data.iloc[:, 1].values
    # values = np.array([2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, -11, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.])
    # values = np.array([1]*10)

    if __name__ == '__main__':
        print(mk.yue_wang_modification_test(values, alpha=0.05, lag=min(min_interval, len(values)//2)))
    
    whole_result = mk.yue_wang_modification_test(values, alpha=0.05, lag=min(min_interval, len(values)//2))
    trend = whole_result.trend
    if trend == 'increasing':
        return {'trend': trend}, f"纵观数据，数据整体呈上涨趋势。"
    elif trend == 'decreasing':
        return {'trend': trend}, f"纵观数据，数据整体呈下跌趋势。"
    else:
        return {}, f"纵观数据，无明显趋势。"

    # seb_sequense trend detection
    upward_pairs = []
    downward_pairs = []
    # find pairs
    for i in range(1, len(values)):
        for j in range(i+min_interval, len(values)):
            sub_sequence = values[i:j]
            # print(f"子序列：{sub_sequence}")
            # print(values[i], values[j])
            if values[i] < values[j]:
                if min(sub_sequence) == values[i] and max(sub_sequence) == sub_sequence[-1]:
                    upward_pairs.append((i,j))
            elif values[i] > values[j]:
                if max(sub_sequence) == values[i] and min(sub_sequence) == sub_sequence[-1]:
                    downward_pairs.append((i,j))

    # print(upward_pairs)

    # print(downward_pairs)

    increasing_pairs = []
    decreasing_pairs = []
    # find increasing and decreasing pairs
    for pair in upward_pairs:
        sub_sequence = values[pair[0]:pair[1]]
        result = mk.yue_wang_modification_test(sub_sequence, alpha=0.01, lag=(len(sub_sequence)//2))
        if result.trend == 'increasing':
            increasing_pairs.append(pair)

    for pair in downward_pairs:
        sub_sequence = values[pair[0]:pair[1]]
        result = mk.yue_wang_modification_test(sub_sequence, alpha=0.01, lag=len(sub_sequence)//2)
        # result = mk.original_test(sub_sequence)
        if result.trend == 'decreasing':
            decreasing_pairs.append(pair)

    final_increasing_pairs = []
    final_decreasing_pairs = []
    # filter pairs

    for pair in increasing_pairs:
        sub_sequence = values[pair[0]:pair[1]]
        print(f"上涨趋势序列：{sub_sequence}")

    for pair in decreasing_pairs:
        sub_sequence = values[pair[0]:pair[1]]
        print(f"下跌趋势序列：{sub_sequence}")


if __name__ == '__main__':
    api_return="""{\"content\":{\"names\":[\"\\u5fae\\u535a\"],\"plot_type\":\"line\",\"title\":\"\\u58f0\\u91cf\\u8d8b\\u52bf\",\"x_title\":\"\\u6c7d\\u8f66\",\"xvalues\":[\"2024-01-01\",\"2024-01-02\",\"2024-01-03\",\"2024-01-04\",\"2024-01-05\",\"2024-01-06\",\"2024-01-07\",\"2024-01-08\",\"2024-01-09\",\"2024-01-10\",\"2024-01-11\",\"2024-01-12\",\"2024-01-13\",\"2024-01-14\",\"2024-01-15\",\"2024-01-16\",\"2024-01-17\",\"2024-01-18\",\"2024-01-19\",\"2024-01-20\",\"2024-01-21\",\"2024-01-22\",\"2024-01-23\",\"2024-01-24\",\"2024-01-25\",\"2024-01-26\",\"2024-01-27\",\"2024-01-28\",\"2024-01-29\",\"2024-01-30\",\"2024-01-31\",\"2024-02-01\",\"2024-02-02\",\"2024-02-03\",\"2024-02-04\",\"2024-02-05\",\"2024-02-06\",\"2024-02-07\",\"2024-02-08\",\"2024-02-09\",\"2024-02-10\",\"2024-02-11\",\"2024-02-12\",\"2024-02-13\",\"2024-02-14\",\"2024-02-15\",\"2024-02-16\",\"2024-02-17\",\"2024-02-18\",\"2024-02-19\",\"2024-02-20\",\"2024-02-21\",\"2024-02-22\",\"2024-02-23\",\"2024-02-24\",\"2024-02-25\",\"2024-02-26\",\"2024-02-27\"],\"y_title\":\"\\u5e74\\u4efd\",\"yvalues\":[[4683,6346,5198,3815,5484,3100,5649,777,2025,740,2781,3511,464,535,2265,3917,441,513,1479,1931,425,1144,2401,1424,2697,1663,4292,1919,2860,7074,3034,4745,2298,3840,885,1119,755,642,663,780,255,376,484,552,392,2266,6393,341,1163,7990,3199,1262,833,891,1059,810,1352,0]]},\"isTab\":true,\"type\":\"chart\"}"""
    api_return = json.loads(api_return)
    api_return['content']['xvalues'].pop()
    api_return['content']['yvalues'][0].pop()
    data = DataFrame({'date': api_return['content']['xvalues'], 'value': api_return['content']['yvalues'][0]})
    print(data)
    # values = pd.Series([1,2,1,2,5,6,4]* 5)
    # values = pd.Series([1,2,1,2,3,4, 2]* 5 + [50] + [1,2,1,2,3,4, 2]* 5)
    # values[15]-=100
    # dates = pd.date_range(start='2021-01-01', periods=len(values), freq='D')
    # data = DataFrame({'date': dates, 'value':values})
    # print(outlier_detection(data, image_path='pic/outlier_detection_1.png'))

    # print(outlier_detection(data, image_path='pic/outlier_detection_2.png'))
    print(trend_detection(data, min_interval=7))
    # print(seasonality_detection(data,threshold=0.05,period=7))


