"""
Author: Buddhi W
Date: 07/31/2024
Main execution script for the AI assistant
"""

import requests
from dotenv import load_dotenv
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils import read_txt_files_in_folder, split_text_data, format_docs
from config.read_config import read_config

load_dotenv()

## Setup the API keys
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

def build_RAG_pipeline(data_folder):

    ## Reading the text files
    text = read_txt_files_in_folder(data_folder)

    ## Split data into chunks
    docs = split_text_data(text, chunk_size=1000, chunk_overlap=200)

    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    return retriever

def assistant(question):
    
    cfg = read_config('config/config.ini')

    retriever = build_RAG_pipeline(cfg['rag_path'])

    ## Template based on hub.pull("rlm/rag-prompt")
    template = """Use the following pieces of context to answer the questions related to interior and exterior design of homes. Please respond without using double-quotation marks. 
    If the question is not related to interior or exterior design, politely say that your are an assistant helping with interior and exterior design and tell the user to ask relavant questions, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=cfg['temperature'], model_name=cfg['model_name'])

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
    )

    out = rag_chain.invoke(question)
    return out


out = assistant("What color scheme should I use in a model kitchen?")
print(out)









