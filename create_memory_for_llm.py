import os

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "F:\\Rag_chatbot\\data\\ANATOMY_PHYSIOLOGY.pdf"

DB_FAISS_PATH = "vectorstore/db_faiss"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# LOAD PDF FILES
# =========================================================

def load_pdf_files():
    pdf_path = r"F:\\Rag_chatbot\\data\\ANATOMY_PHYSIOLOGY.pdf"

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    print(f"Loading PDF: {os.path.basename(pdf_path)}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Successfully loaded {len(documents)} pages.")

    return documents


# =========================================================
# CREATE TEXT CHUNKS
# =========================================================

def create_chunks(documents):

    print("\nCreating text chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    return chunks


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

def get_embedding_model():

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


# =========================================================
# CREATE FAISS DATABASE
# =========================================================

def create_vector_database():

    documents = load_pdf_files()

    chunks = create_chunks(
        documents
    )

    embeddings = get_embedding_model()

    print("\nCreating FAISS vector database...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    os.makedirs(
        os.path.dirname(DB_FAISS_PATH),
        exist_ok=True
    )

    vectorstore.save_local(
        DB_FAISS_PATH
    )

    print("\n====================================")
    print("FAISS DATABASE CREATED SUCCESSFULLY")
    print("====================================")

    print(
        f"Location: {DB_FAISS_PATH}"
    )

    print(
        f"Total chunks: {len(chunks)}"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    create_vector_database()