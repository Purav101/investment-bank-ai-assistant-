import os 
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader

model_name ="BAAI/bge-large-en"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings':False}
embeddings =  HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs 
)
loader = DirectoryLoader('data/',glob = "**/*.pdf",show_progress=True,loader_cls = PyPDFLoader)
documents = loader.load()
text_spliter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 100)
texts =  text_spliter.split_documents(documents)

vector_store = Chroma .from_documents(texts,embeddings, collection_metadata = {"hnsw : space": "cosine"},
persist_directory = "stores/banking_cosine")
print("Vector storage is created.....")