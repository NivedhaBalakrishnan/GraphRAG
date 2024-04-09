from graphRAG import GraphRAG


class Evaluation:
    def __init__(self):
        self.model = GraphRAG()
    

    def evaluate(self):
        self.model.get_info()



# 1. 
# query
# retrieved document
# answer

# similary score -> cosine
# qualitative evaluation -> COT prompt


# csv, unique id, question, source, answer, similarity score, qualitative evaluation