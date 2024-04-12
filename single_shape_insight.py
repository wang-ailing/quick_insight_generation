import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from typing import List
from type_collections import Insight
from type_collections import create_ae_shape_list
from decomposition_functions import group_by_last_week, breakdown_by_one_dimension
import itertools
from single_shape_detector import daily_radar_volume_single_point_detector


if __name__ == '__main__':
    # TODO: decmoposition by week
    
    daily_radar_volume = pd.read_csv('daily_radar_volume.csv')
    daily_radar_volume['date'] = pd.to_datetime(daily_radar_volume['date'])

    
    ae_list = create_ae_shape_list(
        df=daily_radar_volume,
        date_dimension='date',
        subspace_demision_list=['product_name', 'platform','sentiment', 'volume_type'],
        measure_list=['volume'],
        aggregate_function_list=['sum'],
        start_date=pd.to_datetime('2021-01-01'),
        end_date=pd.to_datetime('2021-01-07'),
        date_length=7,
    )



    for ae in ae_list:
        print("-----------------------------------------------------------")
        print(ae.subspace)
        df4single_shape = ae.subspace.reset_index(inplace=False)
        daily_radar_volume_single_point_detector(df4single_shape)
        print("-----------------------------------------------------------")

    
