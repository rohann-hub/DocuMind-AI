import os
from dotenv import load_dotenv

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Load env
load_dotenv()

#  Load text file
loader = TextLoader("data.txt")
documents = loader.load()

# Split text into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Store in FAISS
db = FAISS.from_documents(docs, embeddings)

# Create retriever
retriever = db.as_retriever()

# Load local LLM (Ollama)
llm = OllamaLLM(model="llama3")

# Helper: format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#  Build prompt properly 
def build_prompt(context, question):
    return f"""
You are a helpful AI assistant.

Answer the question based only on the context below.

Context:
{context}

Question:
{question}

Answer:
"""

# RAG pipeline ()
def rag_pipeline(query):
    docs = retriever.invoke(query)
    context = format_docs(docs)
    prompt = build_prompt(context, query)
    response = llm.invoke(prompt)
    return response

# Chat loop
print("🤖  Chatbot is ready! Type 'exit' to stop.\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    response = rag_pipeline(query)

    print("Bot:", response, "\n")
