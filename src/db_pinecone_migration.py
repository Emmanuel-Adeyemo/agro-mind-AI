import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openai_api_key or not pinecone_api_key:
    raise ValueError("Missing API keys! Please check that OPENAI_API_KEY and PINECONE_API_KEY are set in your .env file.")

ROOT = Path(__file__).resolve().parent.parent
CHROMADB_DIR = ROOT / 'chromadb'

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
local_db = Chroma(persist_directory=CHROMADB_DIR, embedding_function=embeddings)

# extract all docs and meta from chromadb
all_docs = local_db.get(include=['documents', 'metadatas', 'embeddings'])

# connect to pinecone cloud
pc = Pinecone(api_key=pinecone_api_key)
index_name = 'qgen-ai-index'
index = pc.Index(index_name)

# stream in batches
batch_size = 100
for i in range(0, len(all_docs['ids']), batch_size):
    vectors = []
    end_idx = min(i + batch_size, len(all_docs['ids']))

    for j in range(i, end_idx):
        vectors.append({
            "id": all_docs['ids'][j],
            "values": all_docs['embeddings'][j],
            "metadata": {**all_docs['metadatas'][j], "text": all_docs['documents'][j]}
        })

    index.upsert(vectors=vectors)

print("Migration Complete! 245MB ChromaDB successfully shifted to the pinecone cloud.")