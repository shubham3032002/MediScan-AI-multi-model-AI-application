import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
# Load environment variables
load_dotenv(find_dotenv())
DATA_PATH="data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_pdf_files(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Directory {data_path} does not exist.")
    
    loader =DirectoryLoader(data_path,glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    if not documents:
        raise ValueError("No PDF files found in the directory.")
    
    return documents

#srep2 : create text chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_embedding():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Step 4: Store embeddings in FAISS

def store_in_faiss(text_chunks,embedding_model,db_path):
    db=FAISS.from_documents(text_chunks,embedding_model)
    db.save_local(db_path)
    print(f"FAISS database saved at {db_path}")
    
try:
    documents=load_pdf_files(DATA_PATH)
    text_chunks=create_chunks(documents)
    embedding_model=get_embedding()
    store_in_faiss(text_chunks,embedding_model,DB_FAISS_PATH)
    print("processing completed")
    
except Exception as e:
    print(f"error: {e}")    
        

