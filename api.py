from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from rag import stream_ollama_response

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        return StreamingResponse(stream_ollama_response(request.message))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
