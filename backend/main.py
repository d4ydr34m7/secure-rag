from fastapi import FastAPI, Request
from embed_utils import ask_with_context

app = FastAPI()

@app.post("/ask")
async def read_root(request: Request):
    body = await request.json()
    question = body.get("question")
    # ask_with_context() is the main function that does retrieval + answer generation
    response = ask_with_context(question)
    return {"answer": response}