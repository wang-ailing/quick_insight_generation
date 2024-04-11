from insight_analysis import attribution_detection, outstanding_1_detection, outstanding_2_detection, outstanding_last_detection, evenness_detection
from pandas import DataFrame, Series
from type_collections import Insight
from type_collections import InsightType

def daily_radar_volume_single_point_detector(
        subspace,
        breakdown_dimension,
        selected_df_grouped_mean: Series) -> Insight:
    
    # volume = data['volume']
    volume = selected_df_grouped_mean

    # attribution detection
    attribution_result = attribution_detection(volume)
    # print(attribution_result)

    # outstanding 1 detection
    outstanding_1_result = outstanding_1_detection(volume)
    # print(outstanding_1_result)

    # outstanding 2 detection 
    # TODO: 后续处理
    outstanding_2_result = outstanding_2_detection(volume)
    print(outstanding_2_result)

    # outstanding last detection
    # TODO: 后续处理
    outstanding_last_result = outstanding_last_detection(volume)
    print(outstanding_last_result)

    # evenness detection
    # TODO: 后续处理
    evenness_result = evenness_detection(volume)
    print(evenness_result)

    insight_type = None

    index_from_single_point = ""
    string_from_single_point = ""
    if attribution_result[0].get('index') != -1:
        insight_type = InsightType.ATTRIBUTION

        index_from_single_point = attribution_result[0].get('index')
        string_from_single_point = attribution_result[1]
    if attribution_result[0].get('index') != -1 and outstanding_1_result[0].get('index') != -1:
        insight_type = InsightType.ATTRIBUTION_OUTSTANDING_1

        index_from_single_point = attribution_result[0].get('index')
        string_from_single_point = attribution_result[1]
        string_from_single_point += "，并且" + outstanding_1_result[1]
    # elif outstanding_2_detection[0].get('index') != -1:
    elif string_from_single_point == "":
        return None


    # print("subspace:", subspace)

    subspace_used = {key: value for key, value in subspace.items() if value != '*'}
    if len(subspace_used) == 0:
        sentence_for_insight = "总体来看，"
        sentence_for_insight += breakdown_dimension + "中" + str(index_from_single_point) + "的声量" + string_from_single_point + "。"
    else :
        subspace_list = list(subspace_used.items())
        sentence_for_insight = "当"
        for index, (dimension, dimension_value) in enumerate(subspace_list):
        # 构建句子
            sentence_for_insight += dimension + "为" + dimension_value
        # 如果不是最后一个元素，添加逗号和分隔符
            if index < len(subspace_list) - 1:
                sentence_for_insight += "、"
        sentence_for_insight += "时，"
        # for breakdown, breakdown_value in breakdown_dimensions.items():
        sentence_for_insight += breakdown_dimension + "中" + str(index_from_single_point) + "的声量" + string_from_single_point + "。"
        
    # print(sentence_for_insight)
    return Insight(insight_type = insight_type, insight_sentence=sentence_for_insight)
    