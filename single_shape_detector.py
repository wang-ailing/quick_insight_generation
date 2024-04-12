from insight_analysis import change_point_detection, outlier_detection, trend_detection
from pandas import DataFrame

def daily_radar_volume_single_point_detector(
        df: DataFrame,

    ) :

    # change point detection
    change_points = change_point_detection(df)
    print("检测到{}个显著变化点：".format(len(change_points)))
    print(change_points)

    # outlier detection
    outliers = outlier_detection(df)
    # print(outliers)
    print("需要关注的日期和值为：")
    for i in range(len(outliers)):
        if outliers[i] == 1:
            print(df.iloc[i, 0].date(), df.iloc[i, 1])

    # trend detection
    trends = trend_detection(df)
    print(trends)