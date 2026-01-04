import pandas as pd
from sklearn.model_selection import train_test_split  # For stratified sampling
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb  # Or import faiss if using FAISS
from chromadb.utils import embedding_functions
import uuid
import os

# Load filtered data
df = pd.read_csv('../data/processed/filtered_complaints.csv')

# Stratified sample: 12K complaints, proportional by product
sample_df, _ = train_test_split(
    df, test_size=0.8, stratify=df['product'], random_state=42)
print(f"Sample shape: {sample_df.shape}")
print(sample_df['product'].value_counts(normalize=True))  # Check proportions

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)
chunks = []
for idx, row in sample_df.iterrows():
    texts = splitter.split_text(row['cleaned_narrative'])
    for i, text in enumerate(texts):
        chunks.append({
            'id': str(uuid.uuid4()),
            'text': text,
            'complaint_id': row.get('complaint_id', idx),
            'product': row['product'],
            'chunk_index': i,
            'total_chunks': len(texts)
        })

chunks_df = pd.DataFrame(chunks)
print(f"Total chunks: {len(chunks_df)}")

# Embedding Model Choice: all-MiniLM-L6-v2 - lightweight (384 dims), fast, good for semantic similarity on short texts like complaints. Trained on 1B sentence pairs, balances quality/speed.

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks_df['text'].tolist())

# Vector Store: Using ChromaDB for simplicity (persistent, easy metadata)
os.makedirs('../vector_store', exist_ok=True)
chroma_client = chromadb.PersistentClient(path='../vector_store')
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2')

collection = chroma_client.get_or_create_collection(
    name="complaints",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"}
)

# Add to collection (batched)
batch_size = 1000
for i in range(0, len(chunks_df), batch_size):
    batch = chunks_df.iloc[i:i+batch_size]
    ids = batch['id'].tolist()
    texts = batch['text'].tolist()
    metadatas = batch.drop(['id', 'text'], axis=1).to_dict('records')
    collection.add(
        documents=texts,
        embeddings=None,  # Auto-embed
        metadatas=metadatas,
        ids=ids
    )

print(f"Indexed {collection.count()} vectors")
