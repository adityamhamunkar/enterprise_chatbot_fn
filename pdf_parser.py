from docling.document_converter import DocumentConverter
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"