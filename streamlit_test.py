import streamlit as st
import tempfile
import os
import networkx as nx

from docling.loaders import DoclingExcelLoader
from docling.pipelines import EntityExtractionPipeline

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# --- Helper: Load Excel with docling ---
def load_excel_with_docling(file_path):
    loader = DoclingExcelLoader(file_path)
    docs = loader.load()
    return docs  # List[Document]

# --- Helper: Extract entities with docling ---
entity_pipeline = EntityExtractionPipeline()
def extract_entities_docling(text):
    result = entity_pipeline.run(text)
    return [ent['text'] for ent in result['entities']]

# --- Helper: Build knowledge graph ---
def build_knowledge_graph(chunks):
    G = nx.Graph()
    chunk_map = {}
    for i, chunk in enumerate(chunks):
        entities = extract_entities_docling(chunk.page_content)
        chunk_id = f"chunk_{i}"
        chunk_map[chunk_id] = chunk
        for ent in entities:
            G.add_node(ent)
            G.add_edge(ent, chunk_id)
    return G, chunk_map

# --- Helper: Get graph-related chunks for a query ---
def get_graph_related_chunks(query, G, chunk_map):
    entities = extract_entities_docling(query)
    related_chunk_ids = set()
    for ent in entities:
        if ent in G:
            related_chunk_ids.update(n for n in G.neighbors(ent) if n.startswith("chunk_"))
    return [chunk_map[cid] for cid in related_chunk_ids]

# --- Streamlit UI ---
st.title("Excel Graph RAG with docling")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # --- Load and chunk ---
    docs = load_excel_with_docling(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    st.info(f"🔀 Chunked into {len(chunks)} pieces.")

    if len(chunks) == 0:
        st.error("No chunks generated—check your input Excel file.")
        st.stop()

    # --- Build knowledge graph ---
    G, chunk_map = build_knowledge_graph(chunks)
    st.success("Knowledge graph built from Excel entities.")

    # --- Build FAISS vectorstore ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    # --- Query interface ---
    user_query = st.text_input("Ask a question about your Excel data:")

    if user_query:
        # 1. Graph RAG: get related chunks
        graph_chunks = get_graph_related_chunks(user_query, G, chunk_map)
        if graph_chunks:
            st.info(f"Graph RAG: Found {len(graph_chunks)} related chunks via knowledge graph.")
            # Use only these for vector search
            search_chunks = graph_chunks
        else:
            st.info("Graph RAG: No related chunks found, using all chunks.")
            search_chunks = chunks

        # 2. Build a temporary vectorstore for the search set
        temp_vectorstore = FAISS.from_documents(search_chunks, embedding=embeddings)
        # 3. Retrieve top-k relevant chunks
        results = temp_vectorstore.similarity_search(user_query, k=3)
        st.write("Top relevant chunks:")
        for i, doc in enumerate(results, 1):
            st.markdown(f"**Chunk {i}:**\n```\n{doc.page_content}\n```")

    # Clean up temp file
    os.unlink(file_path)