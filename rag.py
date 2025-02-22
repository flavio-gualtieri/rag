import logging
from google.cloud import bigquery
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from tools import Tools  # Assuming the Tools class handles PDF reading, chunking, and embedding storage
import google.generativeai as genai
import json
import pandas as pd

class RAG:
    def __init__(self, tools: Tools, chunks=20, window=10):
        self.gemini_api_key = self.__get_api_key()
        self.chunks = chunks
        self.window = window
        self.tools = tools
        self.client = bigquery.Client()
        self.logger = self.__init_logger()
        self.model_name = "gemini-1.5-flash"
        
        # Initialize Gemini API
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel(self.model_name)


    def __init_logger(self) -> logging.Logger:
        logger = logging.getLogger("RAG_LOGGER")
        return logger


    def __get_api_key(self, filepath: str = "config.txt") -> str:
        with open(filepath, "r") as file:
            for line in file:
                if line.startswith("GEMINI_API_KEY="):
                    return line.strip().split("=")[1]


    def __create_prompt(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(context_chunks)
        return f"""
        ### TASK: You are an AI assistant tasked with providing informed responses to questions regarding a documet.
        Answer based on the context provided.

        ###RESPONSE FORMAT: Respond in clear, full sentences that refer to the context provided.

        ### WARNING: If you are unable to provide an accurate answer, say so. 
        Do not hallucinate answers if you cannto respond accureately.


        ### Context:{context}
        ### Question: {question}
        ### Answer:
        """


    def process_pdf(self, file_path: str, document_name: str) -> pd.DataFrame | None:

        try:
            # Check if document already exists in the database
            if self.tools.document_exists(document_name):
                self.logger.info(f"Document '{document_name}' already exists in the database. Skipping chunking.")
                return None  # No need to process the document again
            
            text, _ = self.tools.pdf_reader(file_path)
            self.logger.info("Extracted text")
        except Exception as e:
            self.logger.warning(f"Failed to obtain text from pdf: {e}")
            return None

        if text:
            try:
                chunk_df = self.tools.text_chunker(text, document_name)
                self.logger.info("Chunked and embedded")
            except Exception as e:
                self.logger.warning(f"Failed to chunk and embed:{e}")
            try:
                self.tools.push_df_to_db(chunk_df, document_name)
                self.logger.info("Pushed to df")
            except Exception as e:
                self.logger.warning(f"Failed to push to db: {e}")
            return chunk_df
        return None


    def retrieve_relevant_chunks(self, question: str, document_name: str, top_k: int = 5) -> list[tuple[str, str, float]]:
        question_vector = self.tools.get_embedding(question)

        # Query BigQuery for stored document chunks
        try:
            query = f"""
            SELECT UUID, CHUNK, EMBEDDING
            FROM `{self.tools.table_ref}`
            WHERE DOCUMENT_NAME = '{document_name}'
            """

            results = self.client.query(query).result()
            self.logger.info("Downloaded data")
        except Exception as e:
            self.logger.warning(f"Failed to download data from table: {e}")

        embeddings = []
        chunk_texts = []
        chunk_ids = []

        for row in results:
            try:
                vector = np.array(row["EMBEDDING"], dtype=np.float32)  # Convert JSON string to NumPy array
                embeddings.append(vector)
                chunk_texts.append(row["CHUNK"])
                chunk_ids.append(row["UUID"])
            except Exception as e:
                print(f"Error processing row {row['UUID']}: {e}")

        if not embeddings:
            return []

        embeddings = np.stack(embeddings)  # Convert list of arrays into a 2D NumPy array

        try:
            similarities = cosine_similarity([question_vector], embeddings)[0]
            top_indices = similarities.argsort()[-top_k:][::-1]
            self.logger.info("Computing cosine similarity")
        except Exception as e:
            self.logger.warning(f"Failed to compute cosine similarity: {e}")
        
        try:
            relevant_chunks = [(chunk_ids[i], chunk_texts[i], similarities[i]) for i in top_indices]
            self.logger.info("Obtained relevant chunks")
        except Exception as e:
            self.logger.warning(f"Failed to retrieve relevant chunks: {e}")

        return relevant_chunks


    def rephrase_question(self, question: str) -> str:
        try:
            response = self.model.generate_content(f"Rephrase the following question for clarity: {question}")
            self.logger.info("Rephrased question")
        except Exception as e:
            self.logger.warning(f"Failed to rephrase question: {e}")

        return response.candidates[0].content.parts[0].text.strip()

    
    def generate_text(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()


    def generate_answer(self, question: str, document_name: str) -> str:
        self.logger.info("Rephrasing question using LLM")
        rephrased_question = self.rephrase_question(question)
        
        relevant_chunks = self.retrieve_relevant_chunks(rephrased_question, document_name)
        
        # Extract text directly
        chunk_texts = [chunk[1] for chunk in relevant_chunks]
        
        prompt = self.__create_prompt(rephrased_question, chunk_texts)
        return self.generate_text(prompt)


import os

# Initialize Tools and RAG
tools = Tools(chunk_size=500, overlap=100)
rag = RAG(tools)

# Define file path and document name
file_path = "/Users/flaviogualtieri/Downloads/fastfacts-what-is-climate-change3.pdf"  # Replace with actual file path
document_name = os.path.basename(file_path)

# Process the PDF and store chunks
rag.process_pdf(file_path, document_name)

# Define question
question = "What greenhouse gasses are names in this text?"  # Replace with actual question

# Generate response
response = rag.generate_answer(question, document_name)

# Print response
print(response)