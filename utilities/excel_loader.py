import os
import tempfile
from llama_index.core import Settings, VectorStoreIndex
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import SimpleDirectoryReader
from langchain.embeddings import HuggingFaceEmbeddings
from ollama import AsyncClient

from llama_index.core.prompts import PromptTemplate

def load_excel_as_query_engine(excel_file, model_name="llama3"):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, excel_file.name)
        with open(file_path, "wb") as f:
            f.write(excel_file.getvalue())

        # Load Excel via DoclingReader
        extractor = DoclingReader()
        loader = SimpleDirectoryReader(
            input_dir=temp_dir,
            file_extractor={".xlsx": extractor}
        )
        documents = loader.load_data()

        # # Configure embedding + LLM
        # Settings.embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
        # async for part in await AsyncClient().chat(model='llama3', messages=[message], stream=True):
        #     chunk = part['message']['content']
        #     output += chunk
        #     yield chunk

        # # Optional: custom chunking parser
        # node_parser = MarkdownNodeParser()

        # # Build index
        # index = VectorStoreIndex.from_documents(documents, transformations=[node_parser])

        # # Set up query engine
        # query_engine = index.as_query_engine(streaming=True)

        # # Optional: custom prompt
        # custom_prompt = PromptTemplate(
        #     "Context information is provided below:\n"
        #     "---------------------\n"
        #     "{context_str}\n"
        #     "---------------------\n"
        #     "Based on the above context, answer the query in a step-by-step, concise, and precise manner. "
        #     "If uncertain, reply with 'I don't know!'.\n"
        #     "Query: {query_str}\n"
        #     "Answer: "
        # )
        # query_engine.update_prompts({"response_synthesizer:text_qa_template": custom_prompt})
        # return query_engine
