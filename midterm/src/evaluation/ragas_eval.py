from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
)

class RAGEvaluator:
    def __init__(self):
        self.metrics = [
            context_recall,
            faithfulness,
            answer_relevancy,
        ]
    
    def evaluate(self, dataset):
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics
        )
        return results 