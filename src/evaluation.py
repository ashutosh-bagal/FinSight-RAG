from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
)
from eval_set import eval_set
from rag import ask, retrieve
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
import os
from ragas.embeddings import LangchainEmbeddingsWrapper

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


def format_results_as_markdown(results):
    markdown_sections = []

    for index, item in enumerate(results, start=1):
        question = item["question"]
        role = item["role"]
        answer = item["answer"]
        contexts = item["contexts"]
        ground_truth = item["ground_truth"]

        context_list = "\n".join(f"- {context}" for context in contexts)

        markdown_sections.append(
            f"""## Evaluation #{index}

- Role: {role}
- Question: {question}
- Ground Truth: {ground_truth}

### Answer
{answer}

### Contexts
{context_list}
"""
        )

    return "\n\n".join(markdown_sections)


results = []

for item in eval_set:
    question = item["question"]
    role = item["role"]
    ground_truth = item["ground_truth"]

    chunks, metadata = retrieve(question, role)
    answer = ask(question, role)

    results.append(
        {
            "role": role,
            "question": question,
            "answer": answer,
            "contexts": chunks,
            "ground_truth": ground_truth,
        }
    )

output_path = Path(__file__).resolve().parent.parent / "evaluation_report.md"
output_path.write_text("", encoding="utf-8")

report = format_results_as_markdown(results)
output_path.write_text(report, encoding="utf-8")


ragas_data = Dataset.from_list(results)

groq_llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))
ragas_llm = LangchainLLMWrapper(groq_llm)

hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

scores = evaluate(
    ragas_data,
    metrics=[context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

df = scores.to_pandas()
print(df)
df.to_csv("ragas_scores.csv", index=False)
