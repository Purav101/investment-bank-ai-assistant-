from flask import Flask,request,jsonify,render_template
import os 
import json 
from langchain_chroma import Chroma
#from ctransformers import AutoModelForCausalLM
#from ctransformers import LLM
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
 
#import ctransformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

app = Flask(__name__)
local_llm ="neural-chat-7b-v3-1.Q4_K_M.gguf"


config = {
    'max_new_tokens': 1024,
    'repetition_penalty': 1.1,
    'temperature': 0.1,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'gpu_layers': 20   # safe for 4GB VRAM, adjust down to 16–18 if needed
}
llm =  CTransformers(
    model=local_llm,   # or a repo ID
    model_type="mistral",
    lib="cuda",
    **config
)

print("llm is initialized for the further process")
prompt_template = """use this  following pieces of information to
 answer the user's question. if you don't try to make up an answer. 
 Context : {context}
 Question : {question}
 only return the helpful answer below and nothing else.
 Helpfull answer:
"""
model_name ="BAAI/bge-large-en"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings':False}
embeddings =  HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs 
)
prompt = PromptTemplate(template = prompt_template,input_variables=['context','question'])
load_vector_store = Chroma(persist_directory = "stores/banking_cosine",embedding_function=embeddings)
#retrieve the embeddings present in the storage 
retriver = load_vector_store.as_retriever(search_kwargs={"k": 1})
@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/get_response',methods = ['POST'])
def get_response():
    query = request.form.get('query')
    #logic for handleing query 
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriver,
        return_source_documents = True,
        chain_type_kwargs = chain_type_kwargs,
        verbose = True 
    )
    response = qa(query)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = {"answer" : answer ,"source_document":source_document,"doc" : doc}
    
    return jsonify (response_data)    


if __name__ == '__main__':
 app.run (debug =True,host = '0.0.0.0',port = 5501)
 
