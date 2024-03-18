from typing import Union, Any, List, Dict, Tuple, Optional
from dotenv import load_dotenv
from enum import Enum
from langchain_community.llms import ollama
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, SeleniumURLLoader, YoutubeLoader, Docx2txtLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents import Document
import streamlit as st
import tiktoken as tk
from pathlib import Path
from rich import print as rprint
import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class LLMProviders(Enum):
    OPENAI = "OPENAI"
    OLLAMA = "OLLAMA"

def create_llm(provider: LLMProviders, api_key: str = "", base_url: str = None, model: str = None, temperature: float = 0.7) -> Union[OpenAI, ollama.Ollama]:
    """Create LLM instances based on the provider."""
    if provider == LLMProviders.OPENAI:
        return OpenAI(model='gpt-4-turbo-preview',temperature=temperature, api_key=api_key)
    elif provider == LLMProviders.OLLAMA:
        return ollama.Ollama(base_url=base_url, model=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def langchain_agent(llm: Union[OpenAI, ollama.Ollama], 
                    with_tools: List[str], 
                    agent_type: AgentType, 
                    task: str) -> None:    
    tools = load_tools(with_tools, llm)
    agent = initialize_agent(tools, llm, agent_type, verbose=True)
    result = agent.run(task)

def load_and_split_document(file) -> List[Document]:
    ext = Path(file).suffix
    rprint(f"File extension: {ext}")
    
    if ext == '.pdf':
        doc = PyPDFLoader(file).load()
    elif ext == '.txt':
        doc = TextLoader(file).load()
    if ext == '.docx':
        doc = Docx2txtLoader(file).load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(doc)
    
    return chunks

def main():
    load_dotenv()
     
    api_key = os.getenv("OPENAI_KEY")
    ollama_model = os.getenv("OLLAMA_MODEL")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    
    llm_ollama = create_llm(LLMProviders.OLLAMA, api_key=api_key,
                     base_url=ollama_base_url, model=ollama_model)   
    llm_oai = create_llm(LLMProviders.OPENAI,
                          api_key=os.getenv("OPENAI_KEY"), model='gpt-4-1106-preview')
    
    st.title("QnA Chatbot")
    file = st.file_uploader("Upload a file for analysis", type=["pdf", "docx", "txt"], accept_multiple_files=False)    
    
    def _clear_history():
        if 'chat_history' in st.session_state:
            del st.session_state['chat_history']
        
    add_file = st.button("Add file", on_click=_clear_history)
    
    if file and add_file:        
        with st.spinner("Reading, chunking, and embedding file ..."):
            bytes_data = file.read()
            file_name = os.path.join(PROJECT_ROOT, 'test_data', file.name)
        
            with open(file_name, 'wb') as f:
                f.write(bytes_data)
        
            chunks = load_and_split_document(file_name)
        
            embeddings = HuggingFaceEmbeddings()
            vector_store = Chroma.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever()
            # chain = RetrievalQA.from_chain_type(llm_ollama, retriever=retriever)
            crc = ConversationalRetrievalChain.from_llm(llm_ollama, retriever=retriever)
            st.session_state.crc = crc
            st.success("File added, chunked and embedded successfully")
        
    question = st.text_input("Ask a question")
    if question:                
        if 'crc' in st.session_state:
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []            
                        
            response = st.session_state.crc.run({'question': question, 'chat_history': st.session_state['chat_history']})
            
            st.session_state['chat_history'].append((question, response))
            
            for prompts in st.session_state['chat_history']:
                st.write(f"Q: {prompts[0]}")
                st.write(f"A: {prompts[1]}")

if __name__ == '__main__':
    main()