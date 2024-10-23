"""
Author: Buddhi W
Date: 10/22/2024
Utility functions
"""
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_txt_files_in_folder(folder_path):
    """
     Function for reading text files containing information for RAG
    """
    all_texts = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    filtered_content = ''.join([char for char in content if char not in ['**','#','##','###']])
                    all_texts.append(filtered_content)
    
    return all_texts

def split_text_data(text, chunk_size=1000, chunk_overlap=200):
    """
    Processing data loaded from the text database.
    Performs text splitting and returns split document objects 
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        add_start_index=True
    )

    ## Converting text data into documents
    docs = text_splitter.create_documents(text)

    return docs

def format_docs(docs):
    """
    Format split documents into a format suitable for the retriever
    """
    return "\n\n".join(doc.page_content for doc in docs)






