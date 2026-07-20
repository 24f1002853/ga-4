from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Chunk(BaseModel):
    chunk_id: str
    text: str


class RequestBody(BaseModel):
    question: str
    chunks: List[Chunk]


def tokenize(text):
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/")
def grounded_qa(req: RequestBody):

    if not req.question.strip():
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.3,
            "answerable": False,
        }

    q_tokens = set(tokenize(req.question))

    best_chunk = None
    best_score = 0

    for chunk in req.chunks:
        c_tokens = set(tokenize(chunk.text))

        if len(q_tokens) == 0:
            score = 0
        else:
            score = len(q_tokens & c_tokens) / len(q_tokens)

        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk is None or best_score == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.3,
            "answerable": False,
        }

    confidence = round(min(0.5 + best_score * 0.5, 0.99), 2)

    return {
        "answer": best_chunk.text,
        "citations": [best_chunk.chunk_id],
        "confidence": confidence,
        "answerable": True,
    }