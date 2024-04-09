from eval import *
import uuid
import os
import csv
class GraphRAG():
###existing code
#def
#def
    def get_evaluated(self, question, source, answer):
            evaluation = Evaluation()
            embeddings_model = OpenAIEmbeddings(api_key='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')
            similarity_score = self.evaluation.evaluate_similarity(
                embeddings_model,
                answer,
                source
            )
            score, reason = self.evaluation.evaluate(question, source, answer)
            coherence_score, coherence_reason = evaluation.evaluate_coherence(question, answer)
            faithfulness_score, faithfulness_reason = evaluation.evaluate_faithfulness(question, answer, source)
            #contextual_precision_score, contextual_precision_reason = evaluation.evaluate_contextual_precision(question, answer, source)
            #contextual_recall_score, contextual_recall_reason = evaluation.evaluate_contextual_recall(question, answer, source)
            hallucination_score, hallucination_reason = evaluation.evaluate_hallucination(question, answer, source)
            toxicity_score, toxicity_reason = evaluation.evaluate_toxicity(question, answer)
            bias_score, bias_reason = evaluation.evaluate_bias(question, answer)
            #ragas_score = evaluation.evaluate_ragas(question, answer, source)

            return (similarity_score,score, reason, coherence_score, coherence_reason, faithfulness_score,
                    faithfulness_reason, hallucination_score, hallucination_reason,
                    toxicity_score, toxicity_reason, bias_score, bias_reason)





if __name__ == '__main__':
    graph = GraphRAG()
    question = 'What is inflammation?'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    answer, source = graph.get_response(question)
    
    # Unpack all metric results
    (similarity_score,relevancy_score, reason, coherence_score, coherence_reason, faithfulness_score,
        faithfulness_reason, hallucination_score, hallucination_reason,
        toxicity_score, toxicity_reason, bias_score, bias_reason) = graph.get_evaluated(question, source, answer)

    unique_id = str(uuid.uuid4())
    

    # Prepare data with all metrics for CSV
    csv_data = [timestamp,unique_id, question, source, answer, similarity_score, relevancy_score, reason,
                coherence_score, coherence_reason, faithfulness_score, faithfulness_reason,
                    hallucination_score, hallucination_reason, toxicity_score,
                toxicity_reason, bias_score, bias_reason]

    csv_file_path = 'metrics.csv'
    headers = ['Timestamp','Unique ID', 'Question', 'Source', 'Answer', 'Similarity score', 'Relevancy Score', 'Reason',
                'Coherence Score', 'Coherence Reason', 'Faithfulness Score',
                'Faithfulness Reason', 'Hallucination Score',
                'Hallucination Reason', 'Toxicity Score', 'Toxicity Reason', 'Bias Score',
                'Bias Reason']

    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(csv_data)

    print(f"Scores have been written to {csv_file_path}")
