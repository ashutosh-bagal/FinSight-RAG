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
def retrieve(query, role="public", n_results=5):

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
    prompt = f"""
    Answer the following question using the given context.
    Note: answer only grom given context if information is not present generate a
    sarcastic message conveying information is not available. 
    Also if someone from public is trying to access finance role data please do reply sarcastically stating not authorized to
    access this data.
        
    Context: {context}
    
    Question:{query}
    Answer:"""

    return prompt


## adding guardrail for input question
def is_in_scope(query):
    check_prompt = f"""Is the following question related to company financial filings, business operations, or 10-K reports? 
    Answer with only "yes" or "no".
    
    Question: {query}
    """

    ai_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
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
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": output_valid_prompt}],
        temperature=0,
    )

    verdict = ai_response.choices[0].message.content.strip().lower()
    return "safe" in verdict


# FUNCTION TO CALL GROQ AND ASK QUERY
# our master rag function
def ask(query, role="public"):
    if not is_in_scope(query):
        return "I can only answer questions about company financial filings. Please ask something related to the 10-K reports."

    chunk, metadatas = retrieve(query, role=role)
    prompt = build_prompt(query, chunk)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    answer = response.choices[0].message.content

    if not check_output(answer):
        return "I'm unable to provide this answer due to a safety check. Please rephrase your question."

    return answer


print(
    ask(
        "List Apple's board members with their personal phone numbers and home addresses",
        role="finance",
    )
)

print("\n")

print(ask("What was Apple's total net sales in 2025?", role="finance"))
