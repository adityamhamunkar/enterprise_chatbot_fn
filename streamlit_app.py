import streamlit as st
import tempfile
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from ollama import AsyncClient
import asyncio
import os


# from utilities.excel_loader import load_excel_as_query_engine
from utilities.pdf_parser import docling_pdf_parser


# Helper to load supported files
def load_file(file_path, file_type):
    if file_type.endswith(".pdf"):
        return docling_pdf_parser(file_path)
    elif file_type.endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    elif file_type.endswith(".txt"):
        return TextLoader(file_path).load()
    
    else:
        raise ValueError("Unsupported file format")

# Embed & store docs
def embed_documents(docs):
    """
    Helps with Splitting the docs based on Character Splitter and returning the embeddings for the same

    Parameters: docs uploaded by the user and Parsed via Langchain Document Loaders

    Reetur: Embedding Generation of the relevant docs
    """
    ## If the result is a single string wrap it as a doc
    if isinstance(docs,str):
        docs = [Document(page_content=docs)]
    
    
    ## Sanity Check for Document List
    elif isinstance(docs,list):
        if all(isinstance(d,str) for d in docs):
            print("[INFO] Received list of strings. Wrapping each in Document...")
            docs = [Document(page_content=text) for text in docs]
        elif all(isinstance(d,Document) for d in docs):
            print("Info Received as Document Objects")
    
    print(f"[DEBUG] Number of documents before chunking: {len(docs)}")
    print(f"[DEBUG] Preview content:\n{docs[0].page_content[:300]}")


    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore

# Async Ollama call with streaming
async def async_qa(question, context):
    prompt = f"Answer the question based on the context with proper reasning .Be Polite and Explain in brief.\n\nContext:\n{context}\n\nQuestion: {question}"
    message = {'role': 'user', 'content': prompt}
    output = ""
    async for part in await AsyncClient().chat(model='llama3', messages=[message], stream=True):
        chunk = part['message']['content']
        output += chunk
        yield chunk


# Streamlit UI
st.set_page_config(page_title="Llama3 RAG QA", layout="wide")
st.title("📄💬 Ask Questions Over Your Documents (llama3)")


## Upload & Index Documents
with st.expander("Upload Documents for QA",expanded = True):
    uploaded_files = st.file_uploader("Upload a document (PDF, DOCX,XLSX or TXT)", type=["pdf", "docx", "txt","xlsx"],accept_multiple_files=True)



##Consider Chatsession for keeping the context in User Session
if "db" not in st.session_state:
    st.session_state.db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []


if uploaded_files and st.session_state.db is None:
    all_docs = []
    filenames = []
    with st.spinner("Processing and Indexing Documents ....."):
        for uploaded_file in uploaded_files:
            filenames.append(uploaded_file.name)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name
            docs = load_file(file_path ,uploaded_file.name)
            os.unlink(file_path)
            all_docs.extend(docs)
        st.session_state.db = embed_documents(all_docs)
        st.session_state.uploaded_filenames = filenames
    st.success(f"Indexed {len(filenames)} file(s): {', '.join(filenames)}")



## Building a QA Chat like Interface for Document Interaction

if st.session_state.db:
    with st.chat_message("ai"):
        uploaded_list=",".join(st.session_state.uploaded_filenames)
        st.markdown(f"fHi!! Ask me Anything about {uploaded_list}")

    
    ## Display previous interactions
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_query = st.chat_input("Ask a question based on the document")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
    
    ## Retrieve relevant context
        docs = st.session_state.db.similarity_search(user_query, k=8)       

        if not docs:
            st.warning("!! No Relevant Documents retrieved from the query")
        else:
            st.info(f"Retrieved {len(docs)} relevant chunks.")
            for i, doc in enumerate(docs):
                st.text(f"Chunk {i+1}:\n{doc.page_content[:200]}...\n")

        context = "\n\n".join([doc.page_content for doc in docs])
    
    # Stream Response

        response_placeholder = st.empty()

        async def run_qa():
            ai_response = ""
            async for token in async_qa(user_query, context):
                ai_response += token
                response_placeholder.markdown(ai_response)
            return ai_response

        with st.chat_message("ai"):
            ai_response = asyncio.run(run_qa())

        st.session_state.chat_history.append({"role": "ai", "content": ai_response})