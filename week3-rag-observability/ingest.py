import os
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

FAISS_PATH = "faiss_db"
DOCS_PATH = "docs"


def load_documents():
    docs = []
    for fname in os.listdir(DOCS_PATH):
        fpath = os.path.join(DOCS_PATH, fname)
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(fpath)
        elif fname.endswith(".txt"):
            loader = TextLoader(fpath, encoding="utf-8")
        else:
            continue
        loaded = loader.load()
        docs.extend(loaded)
        print(f"Loaded: {fname} ({len(loaded)} pages)")
    return docs


def ingest():
    print("Loading documents...")
    docs = load_documents()
    if not docs:
        print("No documents found in docs/ folder. Add PDF or TXT files and retry.")
        return

    print(f"Splitting {len(docs)} pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    print(f"Created {len(chunks)} valid chunks")

    print("Generating embeddings...")
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        task_type="retrieval_document"
    )

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    embedded_vecs = embeddings_model.embed_documents(texts)
    print(f"Generated {len(embedded_vecs)} embeddings for {len(texts)} chunks")

    # Build FAISS index manually to avoid batch mismatch bug
    dimension = len(embedded_vecs[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embedded_vecs, dtype=np.float32))

    docstore = InMemoryDocstore({
        str(i): Document(page_content=texts[i], metadata=metadatas[i])
        for i in range(len(texts))
    })
    index_to_docstore_id = {i: str(i) for i in range(len(texts))}

    vectorstore = FAISS(embeddings_model, index, docstore, index_to_docstore_id)
    vectorstore.save_local(FAISS_PATH)
    print(f"Ingestion complete. Vector store saved to {FAISS_PATH}/")


if __name__ == "__main__":
    ingest()
