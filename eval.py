from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric,GEval, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric, HallucinationMetric, ToxicityMetric, BiasMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from datetime import datetime
from scipy.spatial.distance import cosine


class Evaluation:
    def __init__(self, model_name="gpt-4"):
        self.model_name = model_name
        
    def evaluate(self, question,source,answer):
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
        retrieval_context = [source] if source is not None else None
        test_case = LLMTestCase(
            input= question,
            actual_output= answer,
            retrieval_context= retrieval_context
        )
        print("AnswerRelevancyMetric Score and Reason")
        evaluate([test_case], [answer_relevancy_metric])
        return answer_relevancy_metric.score,answer_relevancy_metric.reason
    
    def cosine_similarity(self, vec1, vec2):
        if all(v == 0 for v in vec1) or all(v == 0 for v in vec2):
            return 0.0
        return 1 - cosine(vec1, vec2)

    def evaluate_similarity(self, embeddings_model, answer,source):
        vec1 = embeddings_model.embed_query(answer)
        vec2 = embeddings_model.embed_query(source)
        if not vec1 or not vec2:
            raise ValueError("One of the embedding vectors is empty.")
        return self.cosine_similarity(vec1, vec2)
    
    def evaluate_coherence(self, input_text, actual_output):
        # Define coherence evaluation logic here
        coherence_metric = GEval(
            name="Coherence",
            criteria="Determine if the actual output is coherent with the input.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        )
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output
        )
        coherence_metric.measure(test_case)
        return coherence_metric.score, coherence_metric.reason
    def evaluate_faithfulness(self, input_text, actual_output, retrieval_context):
        metric = FaithfulnessMetric(
            threshold=0.7,
            model="gpt-4",  
            include_reason=True
        )
        retrieval_context = [retrieval_context] if retrieval_context is not None else None
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        metric.measure(test_case)
        return metric.score, metric.reason

    # def evaluate_contextual_precision(self, input_text, actual_output, expected_output, retrieval_context):
    #     metric = ContextualPrecisionMetric(
    #         threshold=0.7,
    #         model="gpt-4",
    #         include_reason=True
    #     )
    #     test_case = LLMTestCase(
    #         input=input_text,
    #         actual_output=actual_output,
    #         expected_output=expected_output,
    #         retrieval_context=retrieval_context
    #     )
    #     metric.measure(test_case)
    #     return metric.score, metric.reason

    # def evaluate_contextual_recall(self, input_text, actual_output, expected_output, retrieval_context):
    #     metric = ContextualRecallMetric(
    #         threshold=0.7,
    #         model="gpt-4",
    #         include_reason=True
    #     )
    #     test_case = LLMTestCase(
    #         input=input_text,
    #         actual_output=actual_output,
    #         expected_output=expected_output,
    #         retrieval_context=retrieval_context
    #     )
    #     metric.measure(test_case)
    #     return metric.score, metric.reason

    def evaluate_hallucination(self, input_text, actual_output, context):
        retrieval_context = [context] if context is not None else None
        metric = HallucinationMetric(threshold=0.5)
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            context= retrieval_context
        )
        metric.measure(test_case)
        return metric.score, metric.reason

    def evaluate_toxicity(self, input_text, actual_output):
        metric = ToxicityMetric(threshold=0.5)
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output
        )
        metric.measure(test_case)
        return metric.score, metric.reason

    def evaluate_bias(self, input_text, actual_output):
        metric = BiasMetric(threshold=0.5)
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output
        )
        metric.measure(test_case)
        return metric.score, metric.reason

    # def evaluate_ragas(self, input_text, actual_output, expected_output, retrieval_context):
    #     metric = RagasMetric(threshold=0.5, model="gpt-3.5-turbo")
    #     test_case = LLMTestCase(
    #         input=input_text,
    #         actual_output=actual_output,
    #         expected_output=expected_output,
    #         retrieval_context=retrieval_context
    #     )
    #     metric.measure(test_case)
    #     return metric.score


