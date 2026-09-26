from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.rag import ask
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


class UserInput(BaseModel):
    question: str = Field(..., title="The question for the item.")
    role: str = Field(title="whats your role?", default="public")


class Passcodecheck(BaseModel):
    passcode: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask")
async def finsight(question: UserInput):
    answer = ask(question.question, role=question.role)
    return JSONResponse({"answer": answer})


@app.post("/verify_passcode")
async def verify_passcode(payload: Passcodecheck):
    correct = os.getenv("FINANCE_PASSCODE")
    return {"valid": payload.passcode == correct}
