import streamlit as st
import tempfile
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from ollama import AsyncClient
import asyncio
import os

# Helper to load supported files
def load_file(file_path, file_type):
    if file_type.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_type.endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    elif file_type.endswith(".txt"):
        return TextLoader(file_path).load()
    else:
        raise ValueError("Unsupported file format")

# Embed & store docs
def embed_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore

# Async Ollama call with streaming
async def async_qa(question, context):
    prompt = f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {question}"
    message = {'role': 'user', 'content': prompt}
    output = ""
    async for part in await AsyncClient().chat(model='gemma3', messages=[message], stream=True):
        chunk = part['message']['content']
        output += chunk
        yield chunk


# Streamlit UI
st.set_page_config(page_title="Gemma3 RAG QA", layout="wide")
st.title("📄💬 Ask Questions Over Your Documents (Ollama + Gemma3)")

uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
query = st.text_input("Ask a question based on the document:", disabled=uploaded_file is None)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name
    docs = load_file(file_path, uploaded_file.name)
    db = embed_documents(docs)
    os.unlink(file_path)

    if query:
        # RAG: retrieve top-k context
        retriever = db.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        # Stream output from Gemma3
        st.markdown("### 💡 Answer:")
        output_box = st.empty()
        async def display_response():
            output_text = ""
            async for token in async_qa(query, context):
                output_text += token
                output_box.markdown(output_text)
        asyncio.run(display_response())
