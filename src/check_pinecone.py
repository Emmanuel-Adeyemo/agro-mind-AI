import os
from dotenv import load_dotenv
from pinecone import Pinecone


load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# connect to pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = "qgen-ai-index"
index = pc.Index(index_name)

# get stats
stats = index.describe_index_stats()

print("--- Pinecone Index Verification ---")
print(f"Index Name:    {index_name}")
print(f"Total Vectors: {stats['total_vector_count']}")
print(f"Dimension:     {stats['dimension']} (Should be 3072)")
print(f"Namespaces:    {stats['namespaces']}")