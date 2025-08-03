# kg_rag_streamlit.py
# Hybrid KG + RAG Streamlit App using LLaMA 3 (via Ollama), FAISS, NetworkX

import os
import streamlit as st
import tempfile
import networkx as nx
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader, UnstructuredExcelLoader
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from utilities.pdf_parser import docling_pdf_parser

# 1. Set up LLM and Embedder
llm = Ollama(model="llama3")
embedder = HuggingFaceEmbeddings()

# 2. PDF/Excel Loader Helper
@st.cache_data

# def load_docs(file):
#     if file.name.endswith(".pdf"):
#         with open(file.name, "wb") as f:
#             f.write(file.getbuffer())
#         return docling_pdf_parser(file.path)
#     elif file.name.endswith(".xlsx"):
#         with open(file.name, "wb") as f:
#             f.write(file.getbuffer())
#         return UnstructuredExcelLoader(file.name).load()
#     return []


def load_docs(file):
    # Create a temporary file and write contents
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getbuffer())
        temp_path = tmp_file.name

    # Pass the path to your custom parser (e.g., docling_pdf_parser)
    if file.name.endswith(".pdf"):
        return docling_pdf_parser(temp_path)  # ✅ pass the saved path
    elif file.name.endswith(".xlsx"):
        return UnstructuredExcelLoader(temp_path).load()
    return []


# 3. Triplet Extraction from Text
def extract_triplets(text):
    triplets = []
    prompt_template = """Extract all subject-relation-object triplets from the following text:

{text}

Format each triplet as: (subject, relation, object)
Return only triplets."""

    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
    for chunk in chunks:
        try:
            prompt = prompt_template.format(text=chunk)
            response = llm.invoke(prompt)
            lines = response.strip().splitlines()
            for line in lines:
                if line.startswith("(") and line.endswith(")") and "," in line:
                    try:
                        triplet = eval(line)  # optionally replace with ast.literal_eval for safety
                        if isinstance(triplet, tuple) and len(triplet) == 3:
                            triplets.append(triplet)
                    except:
                        continue
        except Exception as e:
            print("Triplet extraction failed:", e)
    return triplets


# 4. Build Knowledge Graph (NetworkX)
G = nx.MultiDiGraph()

def build_graph(triplets):
    for s, r, o in triplets:
        G.add_edge(s, o, label=r)

# 5. KG QA (Prompt based)
def query_kg(question):
    edges = list(G.edges(data=True))[:25]  # limit context
    graph_summary = "\n".join([f"{s} --[{d['label']}]--> {o}" for s, o, d in edges])
    prompt = f"""
You are an assistant with access to the following knowledge graph:
{graph_summary}

Use the above information to answer the question:
{question}
"""
    return llm.invoke(prompt)

# 6. RAG Setup
def setup_rag(docs):
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    db = FAISS.from_documents(chunks, embedder)
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# 7. Streamlit UI
st.title("📚 Hybrid RAG + Knowledge Graph Q&A")

uploaded_file = st.file_uploader("Upload a PDF or Excel file", type=["pdf", "xlsx"])

if uploaded_file:
    st.success("File uploaded. Processing...")
    docs = load_docs(uploaded_file)
    raw_text = "\n".join([doc.page_content for doc in docs])

    with st.spinner("Extracting KG triplets..."):
        triplets = extract_triplets(raw_text)
        build_graph(triplets)

    qa_chain = setup_rag(docs)

    mode = st.selectbox("Choose Answering Mode", ["Hybrid", "RAG Only", "KG Only"])
    question = st.text_input("Ask a question:")

    if st.button("Answer") and question:
        answer = ""
        if mode == "RAG Only":
            answer = qa_chain.run(question)
        elif mode == "KG Only":
            answer = query_kg(question)
        else:
            answer_rag = qa_chain.run(question)
            answer_kg = query_kg(question)
            answer = f"🔹 RAG Answer:\n{answer_rag}\n\n🔸 KG Answer:\n{answer_kg}"

        st.markdown("## 📥 Answer")
        st.write(answer)

# 8. Optional KG Visualization
if st.checkbox("Visualize Knowledge Graph"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(s, o): d['label'] for s, o, d in G.edges(data=True)}, ax=ax)
    st.pyplot(fig)
