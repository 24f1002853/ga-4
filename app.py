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


def tokenize(text: str):
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def split_sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


@app.get("/")
def root():
    return {"status": "ok"}


def process(req: RequestBody):

    question = req.question.strip()

    if not question:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }

    q_tokens = tokenize(question)

    best_sentence = None
    best_chunk = None
    best_score = 0.0

    chunk_scores = []

    for chunk in req.chunks:

        sentences = split_sentences(chunk.text)

        max_chunk_score = 0

        for sentence in sentences:

            s_tokens = tokenize(sentence)

            if not s_tokens:
                continue

            overlap = len(q_tokens & s_tokens)

            score = overlap / max(len(q_tokens), 1)

            if score > best_score:
                best_score = score
                best_sentence = sentence
                best_chunk = chunk

            max_chunk_score = max(max_chunk_score, score)

        chunk_scores.append((max_chunk_score, chunk))

    if best_score < 0.2:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": round(best_score, 2),
            "answerable": False,
        }

    citations = []

    for score, chunk in chunk_scores:
        if score >= best_score * 0.9:
            citations.append(chunk.chunk_id)

    citations = list(dict.fromkeys(citations))

    confidence = round(min(0.55 + best_score * 0.4, 0.98), 2)

    return {
        "answer": best_sentence,
        "citations": citations,
        "confidence": confidence,
        "answerable": True,
    }


@app.post("/")
def qa(req: RequestBody):
    return process(req)


@app.post("/grounded-answer")
def qa2(req: RequestBody):
    return process(req)


@app.post("/answer")
def qa3(req: RequestBody):
    return process(req)


@app.post("/api")
def qa4(req: RequestBody):
    return process(req)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
