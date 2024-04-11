from pandas import DataFrame
from .insight_type import InsightType
from typing import Dict, List, Union

class Insight:
    def __init__(self, insight_type: InsightType, entity: DataFrame = None, trigger_values: Dict = None, insight_sentence: str = None, score: float = None):
        self.entity = entity
        self.insight_type = insight_type
        self.trigger_values = trigger_values
        self.insight_sentence = insight_sentence
        self.score = score

    def __str__(self): # tmp_expression
        return f"Insight: {self.insight_sentence} - {self.insight_type}"
    
    def get_insight_type(self) -> InsightType:
        return self.insight_type
    
    def get_insight_sentence(self) -> str:
        return self.insight_sentence