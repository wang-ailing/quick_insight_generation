from pandas import DataFrame

def breakdown_by_one_dimension(selected_df: DataFrame, dimension: str) -> DataFrame:
    selected_df_grouped = selected_df.groupby([dimension]).agg({'volume': ['sum', 'count', 'mean']})
    selected_df_grouped_sum = selected_df_grouped['volume']['sum']
    selected_df_grouped_count = selected_df_grouped['volume']['count']
    selected_df_grouped_mean = selected_df_grouped['volume']['mean']
    return selected_df_grouped_sum