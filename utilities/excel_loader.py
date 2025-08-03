import pandas as pd
from langchain.schema import Document

def df_to_key_value_text(df):
    """
    Converts a DataFrame to a text format where each row is represented as 
    consecutive lines of 'field: value' pairs.
    """
    records_text = ""
    for _, row in df.iterrows():
        for col in df.columns:
            records_text += f"{col}: {row[col]}\n"
        records_text += "\n"  # Add a blank line to separate records
    return records_text.strip()

def excel_parser(excel_path):
    """Excel df parsing with structured text conversion"""
    df = pd.read_excel(excel_path)

    # Option 1: Combine all rows into one document (recommended for small files)
    text = df_to_key_value_text(df)
    return [Document(page_content=text)]

    # Option 2: Split each row into its own Document (good for large datasets)
    # docs = []
    # for _, row in df.iterrows():
    #     row_text = '\n'.join([f"{col}: {row[col]}" for col in df.columns])
    #     docs.append(Document(page_content=row_text))
    # return docs
