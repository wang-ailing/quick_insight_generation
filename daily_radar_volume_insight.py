import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from typing import List
from type_collections import InsightList, InsightType
from decomposition_functions import group_by_last_week, breakdown_by_one_dimension
import itertools
from single_point_detector import daily_radar_volume_single_point_detector


def daily_radar_volume_week_insight(df: DataFrame) -> InsightList:
    
    # print(df)
    # week_df_list = group_by_last_week(df)

    # dimension_list = ['date', 'product_name', 'platform','sentiment', 'volume_type']
    dimension_list = ['product_name', 'platform','sentiment', 'volume_type']
    dimension_values_dict = {}
    for col in dimension_list:
        dimension_values_dict[col] = df[col].unique().tolist()
        dimension_values_dict[col].append(None)
    

    # print(dimension_values_dict)

    combinations_with_attributes = [dict(zip(dimension_values_dict.keys(), combo)) for combo in itertools.product(*dimension_values_dict.values())]

    combinations_with_attributes_with_none = []

    for combination in combinations_with_attributes:
        if None in combination.values():
            combinations_with_attributes_with_none.append(combination)

    insightlist = InsightList()


    for combination in combinations_with_attributes_with_none:
        
        # print(combination)
        product_name = combination['product_name']
        platform = combination['platform']
        sentiment = combination['sentiment']
        volume_type = combination['volume_type']


        if product_name is None and platform is None and sentiment is None and volume_type is None:
            selected_df = df
        else:
            selected_df = df[((df['product_name'] == product_name) if product_name is not None else True)
                            & ((df['platform'] == platform) if platform is not None else True)
                            & ((df['sentiment'] == sentiment)  if sentiment is not None else True) 
                            & ((df['volume_type'] == volume_type) if volume_type is not None else True)]
            
        

        breakdown_dimensions_list = [key for key, value in combination.items() if value is None]

        len_breakdown_dimensions = len(breakdown_dimensions_list)
        # print(breakdown_dimensions)
        # print(len(breakdown_dimensions))
        for dimension in breakdown_dimensions_list:
            subspace_base = {key: str(value) for key, value in combination.items() if value is not None}
            if len_breakdown_dimensions != 1:
                not_breakdown_dimensions = [key for key in breakdown_dimensions_list if key != dimension]
                for not_breakdown_dimension in not_breakdown_dimensions:
                    subspace_base[not_breakdown_dimension] = '*'
            print(subspace_base)

            selected_df_grouped_mean = breakdown_by_one_dimension(selected_df, dimension=str(dimension))
            print(selected_df_grouped_mean)

            if len(selected_df_grouped_mean) == 0 or len(selected_df_grouped_mean) == 1:
                continue

            insight = daily_radar_volume_single_point_detector(
                subspace_base, 
                dimension, 
                selected_df_grouped_mean)
            if insight is not None:
                insightlist.append(insight)

    
    # print(insightlist)
    print("在过去一周内，有如下的声量特点：")
    insightlist_sentences = insightlist.get_insight_sentences()
    for sentence in insightlist_sentences:
        print(sentence)

    print(len(insightlist_sentences))
    return insightlist


if __name__ == '__main__':
    # test code
    daily_radar_volume = pd.read_csv('daily_radar_volume.csv')
    # print(daily_radar_volume['date'])
    daily_radar_volume['date'] = pd.to_datetime(daily_radar_volume['date'])
    daily_radar_volume_week_insight(daily_radar_volume)
    
