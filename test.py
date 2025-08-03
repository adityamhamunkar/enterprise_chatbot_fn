from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index import GPTVectorStoreIndex

# 1. Initialize the reader (exports to Markdown by default)
reader = DoclingReader()

# 2. Load and parse the Excel document
docs = reader.load_data(file_path="annual_report.xlsx")

# 3. Convert Docling documents into LlamaIndex nodes
parser = DoclingNodeParser()
nodes = parser.get_nodes_from_documents(docs)

# 4. Build a vector index for retrieval
index = GPTVectorStoreIndex(nodes)

# 5. Query in natural language
response = index.as_query_engine().query("What was the revenue growth in 2024?")
print(response)
