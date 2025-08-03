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
from utilities.excel_loader import excel_parser


# Helper to load supported files
def load_file(file_path, file_type):
    if file_type.endswith(".pdf"):
        return docling_pdf_parser(file_path)
    elif file_type.endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    elif file_type.endswith(".txt"):
        return TextLoader(file_path).load()
    elif file_type.endswith(".xlsx"):
        return excel_parser(file_path)
    
    else:
        raise ValueError("Unsupported file format")

def embed_documents(docs):
    """
    Splits docs into chunks, embeds them, and builds a FAISS vectorstore,
    showing a progress bar in Streamlit as it goes.
    """
    # --- Normalize input into List[Document] ---
    if isinstance(docs, str):
        docs = [Document(page_content=docs)]
    elif isinstance(docs, list):
        # If it's a list of raw strings, merge into one Document
        if all(isinstance(d, str) for d in docs):
            merged = "\n".join(docs).strip()
            if not merged:
                raise ValueError("No content to embed after merging string docs.")
            docs = [Document(page_content=merged)]
        # If it's already Documents, leave as-is
        elif all(isinstance(d, Document) for d in docs):
            pass
        else:
            raise ValueError("`docs` must be a list of str or a list of Document objects")
    else:
        raise ValueError("`docs` must be a str or list")

    # --- Debug preview ---
    st.info(f"🔍 Preparing to chunk {len(docs)} document(s).")
    st.text(docs[0].page_content[:300] + ("…" if len(docs[0].page_content) > 300 else ""))

    # --- Chunk documents ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    total_chunks = len(chunks)
    st.info(f"🔀 Chunked into {total_chunks} pieces.")

    if total_chunks == 0:
        raise ValueError("No chunks generated—check your input documents.")

    # --- Initialize embeddings & empty FAISS index ---
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
            # for i, doc in enumerate(docs):
            #     st.text(f"Chunk {i+1}:\n{doc.page_content[:200]}...\n")

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