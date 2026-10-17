import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("Groq API key found:", bool(GROQ_API_KEY))

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please check your .env file."
    )


# Initialize Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=500,
    api_key=GROQ_API_KEY
)


# Test question
question = """
What is artificial intelligence?
Explain it in simple terms.
"""


# Send request
response = llm.invoke(question)


print("\n==============================")
print("GROQ MODEL RESPONSE")
print("==============================\n")

print(response.content)