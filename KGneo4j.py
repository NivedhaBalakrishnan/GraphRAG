import os
from helper import Helper
# Common data processing
import json
import textwrap

from dotenv import load_dotenv 

# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from neo4j import GraphDatabase





class NEO4J_KG():
    def __init__(self):
        load_dotenv('.env', override=True)
        hlp = Helper()
        hlp.clear_log_file()  #to clear history
        self.log = hlp.get_logger()

        NEO4J_URI = os.getenv('NEO4J_URI')
        NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
        
        self.kg = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.OPENAI_ENDPOINT = os.getenv('OPENAI_BASE_URL') + '/embeddings'

        self.VECTOR_INDEX_NAME = 'article_chunk'
        self.VECTOR_NODE_LABEL = 'Chunk'
        self.VECTOR_SOURCE_PROPERTY = 'text'
        self.VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'


    def chunk_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2000,
            chunk_overlap  = 200,
            length_function = len,
            is_separator_regex = False,)
        
        chunks = text_splitter.split_text(text)
        return chunks
    
    
    def create_chunks(self, json_data, filename):
        chunks_with_metadata = []
        chunks = self.chunk_text(json_data['text'])
        self.log.info(f"Text split into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            json_data['text'] = chunk
            json_data['chunkSeqId'] = i
            json_data['chunkId'] = f"{filename}_chunk{i:04d}"
            chunks_with_metadata.append(json_data)
        return chunks_with_metadata
    

    def show_index(self):
        index = self.kg.query("SHOW INDEXES")
        self.log.info(f"Index: {index}")
        return index



    def create_graph_nodes(self, chunks_with_metadata):
        merge_chunk_node_query = """
                                MERGE(mergedChunk:Chunk {chunkId: $chunkParam.chunkId})
                                    ON CREATE SET 
                                        mergedChunk.source = $chunkParam.source,
                                        mergedChunk.authors = $chunkParam.authors, 
                                        mergedChunk.journal = $chunkParam.journal, 
                                        mergedChunk.publicationdate = $chunkParam.publicationdate, 
                                        mergedChunk.summary = $chunkParam.summary, 
                                        mergedChunk.text = $chunkParam.text, 
                                        mergedChunk.chunkSeqId = $chunkParam.chunkSeqId
                                RETURN mergedChunk
                                """
        
        try:
            self.kg.query(merge_chunk_node_query, params={"chunkParam": chunks_with_metadata[0]})
            self.kg.query("""CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
                          FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE""")
            # self.log.info(f"Created graph nodes")

        except Exception as e:
            self.log.error(f"An error occurred: {e}")
            raise e

    
    
    
    def create_constraints(self):
        try:
            self.kg.query("""CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
                            FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE""")
            # self.log.info(f"Created constraint")
        except Exception as e:
            self.log.error(f"Index not created: {e}")
            raise e
    


    def get_number_of_nodes(self):
        node_count_query = """
                            MATCH (n:Chunk)
                            RETURN count(n) as nodeCount
                            """
        node_count = self.kg.query(node_count_query)[0]['nodeCount']
        self.log.info(f"Number of nodes: {node_count}")
        return node_count


    def create_vector_index(self):
        try:
            self.kg.query("""
            CREATE VECTOR INDEX `article_chunk` IF NOT EXISTS
            FOR (c:Chunk) ON (c.textEmbedding) 
            OPTIONS { indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'    
            }}
            """)
            # self.log.info(f"Created vector index")
        except Exception as e:
            self.log.error(f"Vector index not created: {e}")
            raise e

    
    def calculate_embeddings(self):
        try:
            # Create an instance of OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY)

            # Retrieve all Chunk nodes without embeddings
            chunks = self.kg.query("""
                MATCH (chunk:Chunk) WHERE chunk.textEmbedding IS NULL
                RETURN chunk
            """)

            for chunk in chunks:
                # Calculate the embedding for the chunk text
                embedding = embeddings.embed_query(chunk['chunk']['text'])

                # Update the Chunk node with the calculated embedding
                self.kg.query("""
                    MATCH (chunk:Chunk {chunkId: $chunkId})
                    SET chunk.textEmbedding = $embedding
                """, params={"chunkId": chunk['chunk']['chunkId'], "embedding": embedding})

            # self.log.info(f"Calculated embeddings")
        except Exception as e:
            self.log.error(f"Embeddings not calculated: {e}")
            raise e


    def process(self, directory):
        files = os.listdir(directory)
        
        for file in files:
            file_path = os.path.join(directory, file)
            json_data = self.get_json_data(file_path)
            file_name = os.path.basename(file_path.split('.')[-2])
            self.log.info(f"Processing file: {file_name}")
            chunked_with_metadata = self.create_chunks(json_data, file_name)
            # self.log.info(f"Chunked data: {chunked_with_metadata}")
            self.create_graph_nodes(chunked_with_metadata)
            self.create_constraints()
            self.create_vector_index()
            self.calculate_embeddings()
            self.kg.refresh_schema()
        print(self.kg.schema)
        total_node_count = self.get_number_of_nodes()
        self.log.info(f"Total number of nodes: {total_node_count}")
        

    


    def get_json_data(self, file_path):
        try:
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                return json_data
        except FileNotFoundError:
            self.log.error("File not found")
            raise FileNotFoundError
        except Exception as e:
            self.log.error(f"An error occurred: {e}")
            raise e




if __name__ == '__main__':
    kg = NEO4J_KG()
    file_path = 'rpapers_json'
    kg.process(file_path)
    

# from neo4j import GraphDatabase

# uri = "bolt://localhost:7687"
# driver = GraphDatabase.driver(uri, auth=("neo4j", "qwerty_102030"))

# def print_greeting(tx, message):
#     result = tx.run("CREATE (a:Greeting) "
#                     "SET a.message = $message "
#                     "RETURN a.message + ', from node ' + id(a)", message=message)
#     print(result.single()[0])

# with driver.session() as session:
#     session.write_transaction(print_greeting, "hello, world")
# driver.close()
