import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


# =========================================================
# CONFIGURATION
# =========================================================

DB_FAISS_PATH = "vectorstore/db_faiss"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GROQ_MODEL = "openai/gpt-oss-120b"


# =========================================================
# CHECK GROQ API KEY
# =========================================================

if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY not found in .env"
    )


# =========================================================
# LOAD EMBEDDINGS
# =========================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# =========================================================
# LOAD FAISS DATABASE
# =========================================================

print("Loading FAISS database...")

if not os.path.exists(DB_FAISS_PATH):

    raise FileNotFoundError(
        "FAISS database not found. "
        "Run create_memory_for_llm.py first."
    )


vectorstore = FAISS.load_local(
    DB_FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)


# =========================================================
# CREATE RETRIEVER
# =========================================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
    }
)


# =========================================================
# INITIALIZE GROQ
# =========================================================

print("Loading Groq model...")

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.2,
    max_tokens=1000,
    api_key=GROQ_API_KEY
)


# =========================================================
# ASK QUESTION
# =========================================================

def ask_question(question):

    # Retrieve relevant documents
    documents = retriever.invoke(
        question
    )

    if not documents:

        return (
            "I could not find relevant information "
            "in the medical documents."
        )


    # Create context
    context_parts = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
Source: {source}
Page: {page}

Content:
{document.page_content}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # Create prompt
    prompt = f"""
You are MediBot, a medical information
assistant.

Answer the user's question using ONLY
the information contained in the provided
medical documents.

Rules:

1. Do not invent information.
2. Do not use unsupported medical facts.
3. If the answer is not present in the
   documents, say that the information is
   not available in the provided documents.
4. Give a clear and easy-to-understand answer.
5. Do not claim to replace a doctor.
6. For emergencies, advise the user to seek
   immediate professional medical attention.

MEDICAL DOCUMENTS:

{context}


USER QUESTION:

{question}


ANSWER:
"""


    # Ask Groq
    response = llm.invoke(
        prompt
    )

    return response.content


# =========================================================
# COMMAND-LINE TEST
# =========================================================

if __name__ == "__main__":

    print("\nMediBot is ready!")

    while True:

        question = input(
            "\nAsk a question "
            "(type 'exit' to quit): "
        )

        if question.lower() == "exit":

            break

        try:

            answer = ask_question(
                question
            )

            print(
                "\nMediBot:\n"
            )

            print(answer)

        except Exception as e:

            print(
                "\nError:",
                e
            )