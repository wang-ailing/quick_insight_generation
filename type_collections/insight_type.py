
from enum import Enum

class InsightType(Enum):
    ATTRIBUTION_OUTSTANDING_1 = 'attribution_outstanding_1'
    ATTRIBUTION = 'attribution'
    OUTSTANDING_1 = 'outstanding_1'
    OUTSTANDING_2 = 'outstanding_2'
    OUTSTANDING_LAST = 'outstanding_last'
    EVENNESS = 'evenness'
    CHANGE_POINT = 'change_point'
    OUTLIER = 'outlier'
    TREND = 'trend'
    SEASONALITY = 'seasonality'
    CORRELATION = 'correlation'