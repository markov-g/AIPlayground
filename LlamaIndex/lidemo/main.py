from dotenv import load_dotenv
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex
import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def main(url: str) -> None:
    document = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    index = VectorStoreIndex.from_documents(documents=document)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is LlamaIndex?")
    print(response)

if __name__ == "__main__":
    load_dotenv()
    url = "https://docs.llamaindex.ai/en/stable/index.html"
    main(url=url)
    