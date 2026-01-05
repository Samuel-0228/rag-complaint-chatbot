import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any
import os


class RAGPipeline:
    def __init__(self, vector_store_path: str = '../vector_store/full_index'):
        """
        Initialize RAG with pre-built embeddings.
        Assumes ChromaDB index at vector_store_path; builds if missing.
        """
        self.model_name = 'all-MiniLM-L6-v2'
        self.embedding_model = SentenceTransformer(self.model_name)

        # Load pre-built parquet
        parquet_path = '../data/raw/complaint_embeddings.parquet'
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Download complaint_embeddings.parquet to {parquet_path}")

        df_chunks = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df_chunks)} chunks from parquet")

        # Build ChromaDB if not exists
        self.client = chromadb.PersistentClient(path=vector_store_path)
        self.collection_name = "full_complaints"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name),
            metadata={"hnsw:space": "cosine"}
        )

        if self.collection.count() == 0:
            # Prepare data: IDs from complaint_id + chunk_index for uniqueness
            df_chunks['id'] = df_chunks['complaint_id'].astype(
                str) + '_' + df_chunks['chunk_index'].astype(str)
            ids = df_chunks['id'].tolist()
            texts = df_chunks['text'].tolist()
            metadatas = df_chunks.drop(['id', 'text', 'embedding'], axis=1).to_dict(
                'records')  # Drop embedding as Chroma auto-handles
            # List of lists/arrays
            embeddings = df_chunks['embedding'].tolist()

            # Batch add (to avoid memory issues; ~1.37M chunks)
            batch_size = 5000
            for i in range(0, len(df_chunks), batch_size):
                batch_df = df_chunks.iloc[i:i+batch_size]
                batch_ids = batch_df['id'].tolist()
                batch_texts = batch_df['text'].tolist()
                batch_metas = batch_df.drop(
                    ['id', 'text', 'embedding'], axis=1).to_dict('records')
                batch_embs = batch_df['embedding'].tolist()

                self.collection.add(
                    documents=batch_texts,
                    embeddings=batch_embs,  # Use pre-computed
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                print(
                    f"Indexed batch {i//batch_size + 1}/{(len(df_chunks)//batch_size)+1}")

        print(f"Vector store ready with {self.collection.count()} vectors")

        # LLM Setup: Mistral-7B (quantized for efficiency; download on first run)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.generator = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def retrieve(self, query: str, k: int = 5, product_filter: str = None) -> List[Dict[str, Any]]:
        """Embed query and retrieve top-k chunks."""
        query_emb = self.embedding_model.encode([query])
        where_clause = {} if not product_filter else {
            "product_category": {"$eq": product_filter}}

        results = self.collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=k,
            where=where_clause
        )

        retrieved = []
        for i in range(len(results['documents'][0])):
            retrieved.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        return retrieved

    def generate(self, query: str, context: str) -> str:
        """Generate answer using prompt template."""
        prompt_template = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. Base your response only on this context. If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:"""

        full_prompt = prompt_template.format(context=context, question=query)

        # Tokenize and generate
        inputs = self.tokenizer(
            full_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.generator(full_prompt, return_full_text=False)

        answer = outputs[0]['generated_text'].strip()
        return answer

    def rag_query(self, query: str, k: int = 5, product_filter: str = None) -> Dict[str, Any]:
        """Full RAG: Retrieve + Generate."""
        docs = self.retrieve(query, k, product_filter)
        context = "\n\n".join([doc['text'] for doc in docs])
        answer = self.generate(query, context)

        return {
            'answer': answer,
            'sources': docs
        }


# Evaluation Script (run separately or in notebook)
if __name__ == "__main__":
    rag = RAGPipeline()

    # Sample Questions (8 total: mix of single/multi-product, trends/issues)
    eval_questions = [
        "Why are people unhappy with Credit Cards?",
        "What are common issues with Personal Loans?",
        "Compare fraud complaints in Money Transfers vs. Savings Accounts.",
        # Note: Filter by date_received if needed
        "Recent trends in billing disputes for Credit Cards (last year).",
        "How does customer service failure manifest in Savings Accounts?",
        "Top sub-issues for high-interest Personal Loans.",
        "Are there fee-related complaints in Money Transfers?",
        "Proactive fixes for repeated Credit Card approval delays."
    ]

    results = []
    for q in eval_questions:
        product_filter = None  # Auto-detect or manual; e.g., extract "Credit Cards" → filter
        if "Credit Cards" in q:
            product_filter = "Credit Card"
        elif "Personal Loans" in q:
            product_filter = "Personal Loan"
        # ... (add logic for others)

        rag_result = rag.rag_query(q, product_filter=product_filter)
        results.append({
            'question': q,
            'answer': rag_result['answer'][:200] + "...",  # Truncate for log
            'sources': [s['text'][:100] + "..." for s in rag_result['sources'][:2]],
            'quality_score': 4  # Placeholder; manual score 1-5 post-run
        })

    # Output table (print or save to CSV/MD)
    import pandas as pd
    eval_df = pd.DataFrame(results)
    print(eval_df.to_markdown(index=False))
    eval_df.to_csv('../data/processed/eval_results.csv', index=False)
