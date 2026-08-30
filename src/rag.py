import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer


load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# model to embed the query
model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_db = chromadb.PersistentClient(path="chroma_db")
collection = chroma_db.get_collection("finance_filings")


# helper function to retrieve chunks FROM CHROMADB COLLECTIONS
def retrieve(query, role="public", n_results=8):

    if role == "finance":
        where_filter = None
    else:
        where_filter = {"role": "public"}

    chunks = collection.query(
        query_texts=[query], n_results=n_results, where=where_filter
    )
    return chunks["documents"][0], chunks["metadatas"][0]


# HELPER FUNCTION TO GENERATE PROMPT
def build_prompt(query, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""You are a professional financial assistant. Answer factually and neutrally using ONLY the context below. Do not give personal opinions, investment advice, or informal commentary. If the answer isn't in the context, say you don't know.
    In case if role is not authorized to see that information, warmly and professionally mention you're not authorized. Use warm and friendly tone.
...
        
    Context: {context}
    
    Question:{query}
    Answer:"""

    return prompt


## adding guardrail for input question
def is_in_scope(query):
    check_prompt = f"""Classify if a question is about company financial filings, business operations, revenue, earnings, or 10-K reports.

Examples:
Question: What was Apple's revenue last year?
Classification: yes

Question: What are the main risk factors?
Classification: yes

Question: Write me a poem about the ocean
Classification: no

Question: What's the weather today?
Classification: no

Now classify this question. Respond with exactly one word: yes or no.

Question: {query}
Classification:"""

    ai_response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": check_prompt}],
        temperature=0,
    )

    answer = ai_response.choices[0].message.content.strip().lower()

    return "yes" in answer


def check_output(answer):
    output_valid_prompt = f"""Does the following answer contain any of these issues: personal data 
    (SSNs, personal emails, phone numbers), 
    or does it clearly contradict 
    "I don't know" while making up specific numbers not typically found in financial filings? Answer with only "safe" or "unsafe".
    
    Output:{answer}
    """

    ai_response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": output_valid_prompt}],
        temperature=0,
    )

    verdict = ai_response.choices[0].message.content.strip().lower()
    return "safe" in verdict


def rewrite_query(query):
    rewrite_prompt = f"""Decide if this question needs rewriting for better document search. Keep clear factual questions AS-IS. Only rewrite if the phrasing is too casual/vague to match formal financial document language.

Examples:
Question: What products does Apple sell?
Rewritten: What products does Apple sell?

Question: How much did Apple spend on research?
Rewritten: What was Apple's research and development (R&D) expense?

Question: What was Apple's R&D spending in fiscal year 2025?
Rewritten: What was Apple's research and development (R&D) expense in fiscal year 2025?

Question: What are Amazon's risk factors?
Rewritten: What are Amazon's risk factors?

Now handle this question. Return ONLY the rewritten (or unchanged) question, nothing else.

Question: {query}
Rewritten:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": rewrite_prompt}],
        temperature=0,
    )
    rewritten = response.choices[0].message.content.strip()
    return rewritten


# FUNCTION TO CALL GROQ AND ASK QUERY
# our master rag function
def ask_with_context(query, role="public"):
    if not is_in_scope(query):
        return (
            "I can only answer questions about company financial filings. Please ask something related to the 10-K reports.",
            [],
        )

    good_query = rewrite_query(query)
    chunk, metadatas = retrieve(good_query, role=role)
    prompt = build_prompt(query, chunk)

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    answer = response.choices[0].message.content

    if not check_output(answer):
        return (
            "I'm unable to provide this answer due to a safety check. Please rephrase your question.",
            chunk,
        )

    return answer, chunk


def ask(query, role="public"):
    answer, _ = ask_with_context(query, role)
    return answer
