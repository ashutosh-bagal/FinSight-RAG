from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.rag import ask


class UserInput(BaseModel):
    question: str = Field(..., title="The question for the item.")
    role: str = Field(title="whats your role?", default="public")


app = FastAPI()


@app.post("/ask")
async def finsight(question: UserInput):
    answer = ask(question.question, role=question.role)
    return JSONResponse({"answer": answer})
