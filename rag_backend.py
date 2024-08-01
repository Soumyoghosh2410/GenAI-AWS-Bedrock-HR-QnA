import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.llms.bedrock import Bedrock

def hr_index():
#define data
    data_load = PyPDFLoader('https://bhel.com/sites/default/files/CDA%20Rules_0.pdf')
    #split data
    data_split = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n", " ", ""], chunk_size = 100, chunk_overlap = 10)

    #create embeddings -- client connection
    data_embeddings = BedrockEmbeddings(
        credentials_profile_name= ' default',
        model_id= 'amazon.titan-embed-text-v1')

    #create vectorDB, store embeddings and Index for search
    data_index = VectorstoreIndexCreator(
        text_splitter = data_split,
        embeddings = data_embeddings,
        vectorstore_cls= FAISS
    )

    #create index for HR Policy Document
        #create vectorstore index from loaders
    db_index = data_index.from_loaders([data_load])
    return db_index

#function for connecting to bedrock foundation model
def hr_llm():
    llm = Bedrock(
        credentials_profile_name = 'default',
        model_id = 'anthropic.claude-v2',
        model_kwargs = {
        "max_tokens_to_sample": 300,
        "temperature": 0.1,
        "top_p": 0.9})
    return llm

#function to search user prompt, search best match from Vector DB, send both to LLM
def hr_rag_response(index, question):
    rag_llm= hr_llm()
    hr_rag_query= index .query(question=question, llm=rag_llm)
    return hr_rag_query

#backedend done
