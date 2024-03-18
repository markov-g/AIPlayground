from enum import Enum
import os
from typing import List, Union
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
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
from crewai import Agent, Task, Crew, Process
from crewai_tools import WebsiteSearchTool
from openai import OpenAI
from rich import print as rprint
from tools.scraper_tools import ScraperTools

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class LLMProviders(Enum):
    OPENAI = "OPENAI"
    OLLAMA = "OLLAMA"

def create_llm(provider: LLMProviders, api_key: str = "", base_url: str = None, model: str = None, temperature: float = 0.7) -> Union[OpenAI, ollama.Ollama]:
    """Create LLM instances based on the provider."""
    if provider == LLMProviders.OPENAI:    
        # return OpenAI(model_name=model,temperature=temperature, api_key=api_key)
        return OpenAI(api_key=api_key)
    elif provider == LLMProviders.OLLAMA:
        return ollama.Ollama(base_url=base_url, model=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

class NewsletterCrew:
  def __init__(self, urls):
    self.urls = urls

  def run(self):
    search_tool = ScraperTools()
          
    # Define Agent
    scraper = Agent(
        role="Summarizer of websites",
        goal="Ask the user for a list of URLs, then go to each given website, scrape the content and provide the full content to the writer angent so it can then be summarized.",
        backstory="""You work at a leading tech think tank. Your expertise lies in taking URLs and getting just the text based content of them.""",
        verbose=True, 
        allow_delegation=False, 
        tools=[search_tool]
    )
    
    writer = Agent(
        role='Tech Content Summarizer and Writer',
        goal='Craft compelling short-form content on AI advancements based on long-form text passed to you',
        backstory="""You are a renowned Content Creator, known for your insightful and engaging articles. You transform complex concepts into compelling narratives.""",
        verbose=True,
        allow_delegation=True
    )
    
    # Create tasks for your agents
    task1 = Task(
        description=f"""Take a list of websites that contain AI content, and then pass it to the writer agent.
        Here are the URLs from the user that you need to scrape: {self.urls}
        """,
        agent=scraper
    )
    
    task2 = Task(
        description="""Using the text provided by the scraper agent, develop short and compelling / interesting summary of the text provided to you about AI.""",
        agent=writer
    )
    
    # Create a crew
    newsletter_crew = Crew(
        name="AI Content Creation Crew",
        agents=[scraper, writer],
        tasks=[task1, task2],
        verbose=2
    )
    
    result = newsletter_crew.kickoff()
    
    return result

def main():
    load_dotenv()
    
    # load LLM
    api_key = os.getenv("OPENAI_KEY")
    ollama_model = os.getenv("OLLAMA_MODEL")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    
    llm_ollama = create_llm(LLMProviders.OLLAMA, api_key=api_key,
                     base_url=ollama_base_url, model=ollama_model)   
    llm_oai = create_llm(LLMProviders.OPENAI,
                          api_key=os.getenv("OPENAI_KEY"), model='gpt-4-1106-preview')   
    
    rprint('''### Welcome to Newsletter Crew!''')
    rprint('-------------------------------------')
    urls = input(""" What is the URL you want to summarize? """)
    
    newsletter_crew = NewsletterCrew(urls=urls)
    result = newsletter_crew.run()
    rprint(result)
    
if __name__ == "__main__":
    main()