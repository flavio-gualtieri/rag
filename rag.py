import logging
import random
import re
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tools import Tools  # Assuming the Tools class handles PDF reading, chunking, and embedding storage

class RAG:
    def __init__(self, tools: Tools, chunks=20, window=10, db_path="embeddings.db"):
        self.chunks = chunks
        self.window = window
        self.tools = tools
        self.db_path = db_path
        self.logger = self.__init_logger()
        self.model_name = "tiiuae/falcon-7b-instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
            device_map="auto"
        )

    def __init_logger(self):
        logger = logging.getLogger("RAG_LOGGER")
        return logger

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

        embeddings = {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in data}
        chunk_ids, vectors = zip(*embeddings.items())
        vectors = np.array(vectors)
        
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
        """Constructs a prompt with context for the LLM."""
        context = "\n\n".join(context_chunks)
        return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    def rephrase_question(self, question: str) -> str:
        """Rephrases the question using the LLM."""
        input_text = f"Rephrase the following question for clarity: {question}"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_length=50)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_length=300)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
