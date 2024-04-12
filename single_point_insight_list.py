import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from typing import List
from type_collections import Insight
from decomposition_functions import group_by_last_week
import itertools
from single_point_detector import point_inght_detector
from type_collections import create_ae_point_list, fetch_ae_point_df

def single_point_insight_list(
        df: DataFrame,
        subspace_dimension_list: List[str],
        measure_list: List[str],
        aggregate_function_list: List[str],
        ) -> List[Insight]:

    insightlist = []

    ae_list = create_ae_point_list(
        df = daily_radar_volume,
        subspace_dimension_list = subspace_dimension_list,
        measure_list = measure_list,
        aggregate_function_list = aggregate_function_list
    )
    
    for ae in ae_list:
        # print("--------------------------------------------------------")
        # print(ae.subspace)
        # print("breakdown dimension:", ae.breakdown_dimension)
        df4insight_detector = fetch_ae_point_df(
            df=daily_radar_volume,
            subspace_expression=ae.subspace,
            breakdown_dimension=ae.breakdown_dimension,
            measure=ae.measure,
            aggregate_function=ae.aggregate_function,
        )

        if len(df4insight_detector) == 0 or len(df4insight_detector) == 1:
            continue
        # print(df)
        insight = point_inght_detector(
            subspace_expression=ae.subspace, 
            breakdown_dimension=ae.breakdown_dimension,
            df=df4insight_detector.iloc[:, 0],
            measure=ae.measure,
        )
        if insight is not None:
            insightlist.append(insight)
    

    return insightlist

if __name__ == '__main__':
    # test code
    daily_radar_volume = pd.read_csv('daily_radar_volume.csv')
    # print(daily_radar_volume['date'])
    daily_radar_volume['date'] = pd.to_datetime(daily_radar_volume['date'])

    insightlist = single_point_insight_list(
        df=daily_radar_volume,
        subspace_dimension_list=['product_name', 'platform','sentiment', 'volume_type'],
        measure_list=['volume'],
        aggregate_function_list=['sum']
    )

    print("在过去一周内，有如下的声量特点：")

    for insight in insightlist:
        sentence = insight.insight_sentence
        print(sentence)

    print(len(insightlist))
