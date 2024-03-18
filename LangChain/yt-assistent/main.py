from enum import Enum
from typing import Union
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.llms import ollama
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import os, argparse

class LLMProviders(Enum):
    OPENAI = "OPENAI"
    OLLAMA = "OLLAMA"

def create_llm(provider: LLMProviders, api_key: str = "", base_url: str = None, model: str = None, temperature: float = 0.7) -> Union[OpenAI, ollama.Ollama]:
    """Create LLM instances based on the provider."""
    if provider == LLMProviders.OPENAI:
        return OpenAI(temperature=temperature, api_key=api_key)
    elif provider == LLMProviders.OLLAMA:
        return ollama.Ollama(base_url=base_url, model=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def create_vector_db_from_youtube(video_url: str, embeddings: OpenAIEmbeddings) -> Chroma:
    """Extracts transcripts from a YouTube video and creates a vector database."""
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    return Chroma.from_documents(docs, embeddings)

def get_response_from_query(db: Chroma, query: str, llm: Union[OpenAI, ollama.Ollama], key: int = 4) -> str:
    """Fetches responses based on similarity search and returns a formatted response."""
    docs = db.similarity_search(query=query, k=key)
    docs_content = " ".join(doc.page_content for doc in docs)
    prompt_template = """
        You are a helpful assistant that can answer questions about YouTube videos 
        based on the video's transcript. Answer the following question: {question}
        By searching the following video transcript: {docs}
        Only use the factual information from the transcript to answer the question.
        If you feel like you don't have enough information to answer the question, say "I don't know".
        Your answers should be verbose and detailed.
    """
    chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["question", "docs"], template=prompt_template))
    response = chain.run(question=query, docs=docs_content).replace("\n", "")
    
    return response

def process_video_query(video_url: str, question: str) -> str:
    """Processes a YouTube video to answer a query."""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_KEY")
    ollama_model = os.getenv("OLLAMA_MODEL")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    
    llm = create_llm(LLMProviders.OLLAMA, api_key=api_key,
                     base_url=ollama_base_url, model=ollama_model)
    embeddings = HuggingFaceEmbeddings()  # Assuming you want to use HuggingFace embeddings
    db = create_vector_db_from_youtube(video_url, embeddings)
    response = get_response_from_query(db, question, llm)
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Process a YouTube video to answer a query.")
    parser.add_argument("-v", "--video_url", type=str, required=True, help="The URL of the YouTube video")
    parser.add_argument("-q", "--question", type=str, required=True, help="The question to be answered based on the video's content")
    
    args = parser.parse_args()

    response = process_video_query(args.video_url, args.question)
    print(response)

if __name__ == "__main__":
    main()