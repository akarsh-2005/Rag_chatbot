
import os
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MediBot | Medical AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"


# =========================================================
# 2. CUSTOM CSS - RESPONSIVE FULL-WIDTH INTERFACE
# =========================================================

st.markdown("""
<style>

/* Overall application */
.stApp {
    background-color: #0e1117;
}

/* Main page width and spacing */
.stMainBlockContainer {
    max-width: 1400px !important;
    width: 100% !important;
    padding: 1.5rem clamp(1rem, 4vw, 3rem) 7rem !important;
    margin: 0 auto !important;
}

/* Header */
h1 {
    text-align: center;
    font-size: clamp(32px, 3.5vw, 46px) !important;
    font-weight: 750 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.4rem !important;
    letter-spacing: -0.5px;
}

/* Subtitle */
[data-testid="stCaptionContainer"] {
    text-align: center;
    font-size: 15px !important;
    color: #a7adbb !important;
    margin-bottom: 1.5rem !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    width: 100% !important;
    max-width: 100% !important;
    padding: 1rem 1.25rem !important;
    border-radius: 14px !important;
    margin-bottom: 1rem !important;
}

/* Chat message text */
[data-testid="stChatMessage"] p {
    font-size: 17px !important;
    line-height: 1.75 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    max-width: 1400px !important;
    width: calc(100% - 2rem) !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Input text */
[data-testid="stChatInput"] textarea {
    font-size: 16px !important;
    line-height: 1.5 !important;
}

/* Buttons */
.stButton button {
    border-radius: 10px !important;
    min-height: 42px;
    font-weight: 500;
}

/* Sources */
[data-testid="stExpander"] {
    border-radius: 10px !important;
    margin-bottom: 1rem !important;
}

/* Welcome panel */
.welcome-panel {
    text-align: center;
    padding: clamp(1.5rem, 4vw, 3rem) 1rem;
    margin: 2rem auto;
    max-width: 850px;
    border: 1px solid rgba(130, 140, 160, 0.2);
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.025);
}

.welcome-icon {
    font-size: 46px;
    margin-bottom: 0.8rem;
}

.welcome-title {
    font-size: clamp(22px, 3vw, 30px);
    font-weight: 700;
    margin-bottom: 0.75rem;
}

.welcome-description {
    font-size: 16px;
    line-height: 1.8;
    color: #a7adbb;
}

/* Mobile devices */
@media (max-width: 768px) {
    .stMainBlockContainer {
        max-width: 100% !important;
        padding: 1rem 0.75rem 6rem !important;
    }

    h1 {
        font-size: 30px !important;
    }

    [data-testid="stChatMessage"] {
        padding: 0.8rem !important;
    }

    [data-testid="stChatMessage"] p {
        font-size: 16px !important;
    }

    [data-testid="stChatInput"] {
        width: calc(100% - 1rem) !important;
    }

    .welcome-panel {
        margin: 1rem auto;
        padding: 1.5rem 0.75rem;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. HEADER
# =========================================================

st.title("🩺 MediBot")
st.caption("Medical Document-Based AI Assistant")


# =========================================================
# 4. SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# 5. VALIDATE CONFIGURATION
# =========================================================

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is missing. Please add it to your .env file."
    )
    st.stop()

if not os.path.isdir(DB_FAISS_PATH):
    st.error("FAISS database not found.")
    st.info("Run create_memory_for_llm.py first.")
    st.stop()


# =========================================================
# 6. LOAD EMBEDDINGS
# =========================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# =========================================================
# 7. LOAD FAISS VECTOR DATABASE
# =========================================================

@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()

    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


# =========================================================
# 8. LOAD GROQ MODEL
# =========================================================

@st.cache_resource
def load_llm(api_key):
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1000,
        api_key=api_key,
    )


# =========================================================
# 9. INITIALIZE RAG COMPONENTS
# =========================================================

try:
    vectorstore = load_vectorstore()
    llm = load_llm(GROQ_API_KEY)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

except Exception as exc:
    st.error("Unable to initialize MediBot.")
    st.exception(exc)
    st.stop()


# =========================================================
# 10. WELCOME SCREEN
# =========================================================

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-panel">
        <div class="welcome-icon">🩺</div>
        <div class="welcome-title">
            Welcome to MediBot
        </div>
        <div class="welcome-description">
            Ask a question about your uploaded medical documents.
            MediBot searches your document knowledge base and
            generates an answer using the retrieved information.
            <br><br>
            Type your question in the chat box below to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 11. DISPLAY CHAT HISTORY AND SOURCES
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Restore source references after Streamlit reruns
        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(sources, start=1):
                        st.markdown(
                            f"**Source {i}:** {source['name']}\n\n"
                            f"**Page:** {source['page']}"
                        )


# =========================================================
# 12. CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask your medical question..."
)


# =========================================================
# 13. PROCESS USER QUESTION
# =========================================================

if question:

    # Display and save the user's question
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    try:
        # Retrieve relevant documents
        with st.spinner("🔎 Searching medical documents..."):
            documents = retriever.invoke(question)

        # Prepare document context
        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        if not context.strip():
            context = (
                "No relevant content was retrieved "
                "from the medical documents."
            )

        # Build prompt
        prompt = f"""
You are MediBot, a medical information assistant.

Instructions:
1. Answer using ONLY the information in the supplied context.
2. If the answer is not supported by the context, clearly
   state that the information is not available in the provided
   medical documents.
3. Do not invent medical facts, diagnoses, or treatments.
4. Explain the available information in clear, understandable
   language.
5. Do not claim that you are a doctor.
6. If the user describes an emergency, advise them to seek
   immediate professional medical assistance.

Context:
{context}

Question:
{question}

Answer:
"""

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("🤖 Generating answer..."):
                response = llm.invoke(prompt)
                answer = response.content

            # Ensure the answer can be displayed as text
            if not isinstance(answer, str):
                answer = str(answer)

            st.markdown(answer)

            # Prepare source metadata
            source_list = []

            for document in documents:
                source_list.append({
                    "name": str(
                        document.metadata.get(
                            "source", "Unknown"
                        )
                    ),
                    "page": str(
                        document.metadata.get(
                            "page", "Unknown"
                        )
                    ),
                })

            # Show source references
            if source_list:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(source_list, start=1):
                        st.markdown(
                            f"**Source {i}:** {source['name']}\n\n"
                            f"**Page:** {source['page']}"
                        )

        # Save answer and sources to session history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": source_list,
        })

    except Exception:
        st.error(
            "Sorry, an error occurred while processing your "
            "question. Please check your Groq connection, "
            "model configuration, and FAISS database."
        )
        st.exception(Exception("MediBot request failed"))
