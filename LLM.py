import cohere
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from pathlib import Path
from llama_index import download_loader
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms.cohere import Cohere


# Cohere Client with an API Key
co = cohere.Client('Yqd4wQBcTH1dYkHYiEqOPY7Ac8rdrvrb2Cst2wLX')



# Load the model

cohere_api_key = 'Yqd4wQBcTH1dYkHYiEqOPY7Ac8rdrvrb2Cst2wLX' #keep your api key in the .env file and retrieve it
model = "command" #this is the model name from cohere. Select it that matches with you 
temperature = 0 # It can be range from (0-1) as openai
max_tokens = 256
llm = Cohere(model=model,temperature=0,cohere_api_key=cohere_api_key,max_tokens=max_tokens)



# Load the embeddings
embeddings = CohereEmbeddings(cohere_api_key='Yqd4wQBcTH1dYkHYiEqOPY7Ac8rdrvrb2Cst2wLX')



# Create a service context
service_context = ServiceContext.from_defaults(llm = llm, embed_model=embeddings)



#Load the PDF file
PDFReader = download_loader("PDFReader")
loader = PDFReader()
documents = loader.load_data(file=Path('DA/Note.pdf'))

print("Number of documents loaded:", len(documents))




# Create an index
index = VectorStoreIndex. from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
query_results = query_engine.retrieve("Why is neptune blue?")


# Query with Cohere
def query_with_cohere(context, query):
    prompt = f"Context: {context}\nQuery: {query}\nAnswer:" 
    response = co.generate(
    model='command',
    prompt=prompt,
    max_tokens=50
    )

    return response.generations[0].text

# Use query results as context
context_from_llama_index = query_results # Extract context from query_results
cohere_response = query_with_cohere(context_from_llama_index, "Why is neptune blue?")
print("Cohere's response:", cohere_response)