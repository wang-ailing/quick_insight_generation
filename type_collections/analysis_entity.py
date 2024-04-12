from typing import List, Union
from pandas import DataFrame
from pandas import Timestamp
import itertools


class AnalysisEntity:
    def __init__(
            self, 
            subspace: Union[ DataFrame, dict ], 
            breakdown_dimension: str, 
            measure: str, 
            aggregate_function: str, 
            ):
        
        self.subspace = subspace
        self.breakdown_dimension = breakdown_dimension
        self.measure = measure
        self.aggregate_function = aggregate_function

def create_ae_point_list(
        df: DataFrame, 
        # date_dimension: str,  # TODO: merge ae_point and ae_shape
        subspace_dimension_list: List[str], 
        measure_list: List[str], 
        aggregate_function_list: List[str], 
        # start_date: Timestamp = None, # TODO: decompose date
        # end_date: Timestamp = None,
        # date_length: int = 1
        ) -> List[AnalysisEntity]:
    
    ae_point_list = []

    dimension_values_dict = {}
    for col in subspace_dimension_list:
        dimension_values_dict[col] = df[col].unique().tolist()
        dimension_values_dict[col].append(None)

    combinations_with_attributes = [dict(zip(dimension_values_dict.keys(), combo)) for combo in itertools.product(*dimension_values_dict.values())]

    combinations_with_attributes_with_none = []

    for combination in combinations_with_attributes:
        if None in combination.values():
            combinations_with_attributes_with_none.append(combination)

    for aggregate_function in aggregate_function_list:

        for combination in combinations_with_attributes_with_none:

            for measure in measure_list:

                breakdown_dimensions_list = [key for key, value in combination.items() if value is None]

                len_breakdown_dimensions = len(breakdown_dimensions_list)

                for breakdown_dimension in breakdown_dimensions_list:
                    subspace_expression = {key: str(value) for key, value in combination.items() if value is not None}
                    if len_breakdown_dimensions != 1:
                        not_breakdown_dimensions = [key for key in breakdown_dimensions_list if key != breakdown_dimension]
                        for not_breakdown_dimension in not_breakdown_dimensions:
                            subspace_expression[not_breakdown_dimension] = '*'
                    print(subspace_expression)

                    ae_point_list.append(AnalysisEntity(
                        subspace=subspace_expression, 
                        breakdown_dimension=breakdown_dimension, 
                        measure=measure, 
                        aggregate_function=aggregate_function,
                        ))


    return ae_point_list

def fetch_ae_point_df(
        df: DataFrame, 
        subspace_expression: dict, 
        breakdown_dimension: str, 
        measure: str, 
        aggregate_function: str
) -> DataFrame:
    
    selected_df = df
    condition_df = True

    for subspace_dimension, name in subspace_expression.items():
        if name == '*':
            continue
        condition_df = condition_df & (df[subspace_dimension] == name)
    
    if type(condition_df) == bool:
        pass
    else:
        selected_df = df[condition_df]

    selected_df_grouped = selected_df.groupby([breakdown_dimension]).agg({measure: aggregate_function})
    
    return selected_df_grouped

def create_ae_shape_list(
        df: DataFrame, 
        date_dimension: str, 
        subspace_dimension_list: List[str], 
        measure_list: List[str], 
        aggregate_function_list: List[str], 
        start_date: Timestamp = None,
        end_date: Timestamp = None,
        date_length: int = 1
        ) -> List[AnalysisEntity]:
    
    ae_shape_list = []

    if start_date is not None and end_date is not None:
        date_length = (end_date - start_date).days + 1

    for aggregate_function in aggregate_function_list:
        for breakdown_dimension in subspace_dimension_list:
            for measure in measure_list:
                selected_df = df.groupby([date_dimension, breakdown_dimension]).agg({measure: aggregate_function})

                values = selected_df.index.get_level_values(breakdown_dimension).unique()

                # print(values)

                for value in values:
                    # for example: xs('微博', level='platform')
                    subspace = selected_df.xs(value, level=breakdown_dimension)
                    
                    if len(subspace) != date_length:
                        continue

                    ae = AnalysisEntity(subspace, breakdown_dimension, measure, aggregate_function)
                    ae_shape_list.append(ae)

    return ae_shape_list

# def fetch_ae_shape_df(
#         df: DataFrame, 
#         date_dimension: str, 
#         subspace_dimension_list: List[str], 
#         measure_list: List[str],
# ):
#     pass