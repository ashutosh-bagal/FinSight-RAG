from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer


files = {
    "apple": "data/raw/aapl-20250927.pdf",
    "amazon": "data/raw/amzn-20251231.pdf",
}


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)


# chunking
all_chunks = []

for company, path in files.items():
    loader = PyPDFLoader(path)
    pages = loader.load()
    chunks = text_splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["company"] = company

    all_chunks.extend(chunks)


# role tagging
finance_start_page = {"apple": 28, "amazon": 33}

for chunk in all_chunks:
    company = chunk.metadata["company"]
    page = chunk.metadata["page"]
    threshold = finance_start_page[company]

    if page >= threshold - 1:
        chunk.metadata["role"] = "finance"
    else:
        chunk.metadata["role"] = "public"


# storing embeddings into chromadb

# create/connect to local chroma
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("finance_filings")

# prepare data for chroma
text = [chunk.page_content for chunk in all_chunks]
metadata = [chunk.metadata for chunk in all_chunks]
ids = [f"chunk_{i}" for i in range(len(all_chunks))]

# generating embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(text).tolist()

# store in chromadb
collection.add(
    documents=text,
    embeddings=embeddings,
    metadatas=metadata,
    ids=ids,
)
