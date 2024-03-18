from typing import Union, Any, List, Dict, Tuple, Optional
from dotenv import load_dotenv
from enum import Enum
from langchain_community.llms import ollama
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma
import streamlit as st
import os


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

def main():
    load_dotenv()
     
    api_key = os.getenv("OPENAI_KEY")
    ollama_model = os.getenv("OLLAMA_MODEL")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    
    llm_ollama = create_llm(LLMProviders.OLLAMA, api_key=api_key,
                     base_url=ollama_base_url, model=ollama_model)   
    llm_oai = create_llm(LLMProviders.OPENAI,
                          api_key=os.getenv("OPENAI_KEY"))
    
    st.title("Medium Article Generator")
    topic = st.text_input("Enter the topic of the article")
    # language = st.text_input("Enter the language of the article")
    title_template = PromptTemplate(
        input_variables=["topic"],
        template='Give me a medium article title on "{topic}" in English'
    )
    
    article_template = PromptTemplate(
        input_variables=["title"],
        template='You are an expert on topic and a great teacher. Give me a Medium article for the title "{title}" of around 450 worlds in length. It should summarize the topic and provide a good introduction to it as well as some concrete actionable insights. Please format you reply as a Medium article in markdown inside a code block: ```md [...] ```.'
    )
    
    title_chain = LLMChain(llm=llm_ollama, prompt=title_template, verbose=True)
    article_chain = LLMChain(llm=llm_ollama, prompt=article_template, verbose=True)
    overall_chain = SimpleSequentialChain(chains=[title_chain, article_chain], verbose=True)
    
    if topic:
        # response = llm.invoke(title_template.format(topic=topic, language=language))
        response = overall_chain.invoke(topic)
        st.write(response)

if __name__ == '__main__':
    main()