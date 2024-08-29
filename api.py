from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import ollama

from rag import get_input

app = FastAPI()


# Define the request body structure using Pydantic
class ChatRequest(BaseModel):
    message: str


# POST endpoint with hardcoded model
@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        return StreamingResponse(stream_ollama_response(request.message))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Reusable method with hardcoded model 'llama3.1'
async def stream_ollama_response(user_message: str):
    input = get_input(user_message)

    print(input)

    stream = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': input}],
        stream=True,
    )
    for chunk in stream:
        yield chunk['message']['content']

