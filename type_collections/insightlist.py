from .insight import Insight

class InsightList(list):
    def __init__(self, initial_data=None):
        if initial_data:
            for item in initial_data:
                if not isinstance(item, Insight):
                    raise TypeError(f"All items must be instances of Insight, got {type(item)} instead.")
            super().__init__(initial_data)
        else:
            super().__init__()

    # def get_insights_by_insight_type(self, type_name):
    #     return [insight for insight in self if insight.insight_type == type_name]
    

    def append(self, item):
        if not isinstance(item, Insight):
            raise TypeError(f"All items must be instances of Insight, got {type(item)} instead.")
        super().append(item)

    def get_insight_sentences(self):
        sentences = []
        for insight in self:
            sentences.append(insight.insight_sentence)

        return sentences

    # def extend(self, items):
    #     for item in items:
    #         if not isinstance(item, Insight):
    #             raise TypeError(f"All items must be instances of Insight, got {type(item)} instead.")
    #     super().extend(items)


    # def insert(self, index, item):
    #     if not isinstance(item, Insight):
    #         raise TypeError(f"All items must be instances of Insight, got {type(item)} instead.")
    #     super().insert(index, item)

    # def __setitem__(self, index, item):
    #     if not isinstance(item, Insight):
    #         raise TypeError(f"All items must be instances of Insight, got {type(item)} instead.")
    #     super().__setitem__(index, item)


    # def __add__(self, other):
    #     if not isinstance(other, InsightList):
    #         raise TypeError(f"Can only concatenate InsightList (not '{type(other)}') to InsightList")
    #     return InsightList(super().__add__(other))
    

    # def __iadd__(self, other):
    #     if not isinstance(other, InsightList):
    #         raise TypeError(f"Can only concatenate InsightList (not '{type(other)}') to InsightList")
    #     return InsightList(super().__iadd__(other))
    