from enum import Enum
from typing import Any, Optional, Union, List, Dict
from langchain_community.llms import ollama
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv, find_dotenv
import os 

def load_env():
    _ = load_dotenv(find_dotenv()) # read local .env file

class llm_providers(Enum):
    OPENAI="OPENAI",
    OLLAMA="OLLAMA",
        
def create_llms(providers: llm_providers, 
                api_key: str = "", 
                base_url: str = None, 
                model: str = None, 
                temp: float = 0.7) -> Union[OpenAI, ollama.Ollama]:
    
    if providers == llm_providers.OPENAI:
        llms = OpenAI(temperature=temp, 
                             api_key=api_key)
    elif providers == llm_providers.OLLAMA:
        llms = ollama.Ollama(base_url=base_url,                              
                             model=model, 
                             temperature=temp)
    
    return llms

def get_prompt(prompt: str, input_vars: list[str]) -> PromptTemplate:
    template = PromptTemplate(input_variables=input_vars, template=prompt)
    
    return template

def get_response(llm: Any, prompt_template: str, variables_values: Dict[str, str], output_key: Optional[str]) -> str:
    # Automatically generate input_vars from variables_values keys
    input_vars = list(variables_values.keys())

    # Create the prompt template
    template = PromptTemplate(input_variables=input_vars,
                              template=prompt_template)
    
    # Initialize the LLM chain with the template
    name_chain = LLMChain(llm=llm, prompt=template, output_key=output_key)
    
    # Get the response by passing the variables' values
    response = name_chain(variables_values)
    
    return response

def langchain_agent(llm: Union[OpenAI, ollama.Ollama], 
                    with_tools: List[str], 
                    agent_type: AgentType, 
                    task: str) -> None:    
    tools = load_tools(with_tools, llm)
    agent = initialize_agent(tools, llm, agent_type, verbose=True)
    result = agent.run(task)

if __name__ == "__main__":
    load_env()
    
    llm = create_llms(llm_providers.OLLAMA, 
                      api_key=os.getenv("OPENAI_KEY"),
                      base_url=os.getenv("OLLAMA_BASE_URL"), 
                      model=os.getenv("OLLAMA_MODEL"))
    
    llm_oai = create_llms(llm_providers.OPENAI,
                          api_key=os.getenv("OPENAI_KEY"))
    
    # One-Shot
    prompt_template = "I have a {animal_type} pet and I want a cool name for it. It is {animal_color} in color. Suggest me five cool names for my pet."
    variables_values = {'animal_type': 'dog', 'animal_color': 'gray'}
    # print(get_response(llm, prompt_template, variables_values, output_key="pet_name"))
    
    # Agents
    tools = ["wikipedia", "llm-math"]
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION
    task = "What is the average age of a dog? Multiply the age by 3"
    print(langchain_agent(llm_oai, tools, agent_type, task))
    