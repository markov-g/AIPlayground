import os
import pdb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms import ollama, openai 
from llama_index.embeddings.huggingface import HuggingFaceE
from llama_index.core import download_loader
from llama_index.core.service_context import ServiceContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# # create client and a new collection
# chroma_client = chromadb.EphemeralClient()
# chroma_collection = chroma_client.create_collection("quickstart")

# # define embedding function
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# # load documents
# documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# # set up ChromaVectorStore and load in data
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context, embed_model=embed_model
# )

# # Query Data
# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# display(Markdown(f"<b>{response}</b>"))

def main():
    load_dotenv()
    chroma_client = chromadb.PersistentClient() # Stores stuff in .chroma

if __name__ == "__main__":
    main()
    