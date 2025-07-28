import os
from dotenv import load_dotenv
from uuid import uuid4
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

def load_documents(directory="data/source_docs"):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

def split_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def ingest_documents():
    documents = load_documents()
    for doc in documents:
        chunks = split_text(doc)
        for chunk in chunks:
            # response = client.embeddings.create(
            #     input=chunk,
            #     model="text-embedding-3-small"
            # )
            # embedding = response.data[0].embedding[:2048]  # Optional truncation
            embedding = [0.1] * 2048  # Mock embedding for testing
            index.upsert([
                (
                    str(uuid4()),
                    embedding,
                    {"text": chunk}
                )
            ])
    print("✅ Documents successfully ingested into Pinecone!")

def cleanup_index():
    index.delete(delete_all=True)
    print("🧹 All vectors deleted from the Pinecone index.")

if __name__ == "__main__":
   # ingest_documents()
    cleanup_index()  # Run cleanup to avoid storage charges
