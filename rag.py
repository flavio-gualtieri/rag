import logging
import random
import re
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from tools import Tools  # Assuming the Tools class handles PDF reading, chunking, and embedding storage
import google.generativeai as genai

class RAG:
    def __init__(self, tools: Tools, chunks=20, window=10, db_path="embeddings.db"):
        self.gemini_api_ley = self.__get_api_key()
        self.chunks = chunks
        self.window = window
        self.tools = tools
        self.db_path = db_path
        self.logger = self.__init_logger()
        self.model_name = "gemini-1.5-flash"
        
        # Initialize Gemini API
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def __init_logger(self):
        logger = logging.getLogger("RAG_LOGGER")
        return logger

    def __get_api_key(self, filepath="config.txt"):
        with open(filepath, "r") as file:
            for line in file:
                if line.startswith("GEMINI_API_KEY="):
                    return line.strip().split("=")[1]

    def process_pdf(self, file_path: str):
        text, _ = self.tools.pdf_reader(file_path)
        if text:
            chunk_df = self.tools.text_chunker(text)
            self.tools.push_df_to_db(chunk_df)
            return chunk_df
        return None

    def retrieve_relevant_chunks(self, question: str, top_k=5):
        question_vector = self.tools.embedder.encode([question])[0]
        
        # Retrieve stored embeddings
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, vector FROM embeddings")
        data = cursor.fetchall()
        conn.close()

        # Ensure data is correctly converted from binary
        embeddings = {}
        for row in data:
            if row[1] is not None:
                embeddings[row[0]] = np.frombuffer(row[1], dtype=np.float32)

        if not embeddings:
            return []

        chunk_ids, vectors = zip(*embeddings.items())
        vectors = np.stack(vectors)  # Ensures correct shape

        # Compute cosine similarity
        similarities = cosine_similarity([question_vector], vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        relevant_chunks = [(chunk_ids[i], similarities[i]) for i in top_indices]
        return relevant_chunks


    def generate_answer(self, question: str):
        self.logger.info("Rephrasing question using LLM.")
        rephrased_question = self.rephrase_question(question)
        
        relevant_chunks = self.retrieve_relevant_chunks(rephrased_question)
        
        # Retrieve chunk texts
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        chunk_texts = []
        for chunk_id, _ in relevant_chunks:
            cursor.execute("SELECT vector FROM embeddings WHERE id=?", (chunk_id,))
            result = cursor.fetchone()
            if result:
                chunk_texts.append(result[0].decode("utf-8"))
        conn.close()
        
        prompt = self.__create_prompt(rephrased_question, chunk_texts)
        return self.generate_text(prompt)

    def __create_prompt(self, question, context_chunks):
        context = "\n\n".join(context_chunks)
        return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    def rephrase_question(self, question: str) -> str:
        response = self.model.generate_content(f"Rephrase the following question for clarity: {question}")
        return response.text.strip()

    def generate_text(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()
