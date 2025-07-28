import os
from dotenv import load_dotenv
import openai
from pinecone import Pinecone

load_dotenv()

# Setup API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

def ask_with_context(question):
    embedding = openai.Embedding.create(
        input=question,
        model="text-embedding-3-small"
    )["data"][0]["embedding"]
    
    results = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True,
    )
    
    context = "\n".join([match["metadata"]["text"] for match in results["matches"]])

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the user's question using the provided context only.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    return completion["choices"][0]["message"]["content"]
