import os
from docling.document_converter import DocumentConverter

### Using Docling PDF parser for more accurate parsing

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"


def docling_pdf_parser(pdf_path):
    converter = DocumentConverter()
    result = converter.convert(pdf_path)  
    print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"
    return result.document.export_to_markdown()
