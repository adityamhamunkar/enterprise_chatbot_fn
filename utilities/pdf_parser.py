import os
from docling.document_converter import DocumentConverter


### Using Docling PDF parser for more accurate parsing

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from langchain.schema import Document

def docling_pdf_parser(pdf_path):
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    markdown_text = result.document.export_to_markdown()    
    print("[DEBUG] First 300 characters of parsed text:\n", markdown_text[:300])    
    return [Document(page_content=markdown_text)]

