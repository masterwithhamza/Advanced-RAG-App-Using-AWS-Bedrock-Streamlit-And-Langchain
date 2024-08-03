import json
import os
import sys
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Setup Bedrock Client
bedrock = boto3.client(service_name="bedrock-runtime")

# Ensure model IDs are correct and available
embedding_model_id = "amazon.titan-embed-text-v1"
llm_claude_model_id = "amazon.titan-text-lite-v1"
llm_llama2_model_id = "meta.llama3-70b-instruct-v1:0"

try:
    bedrock_embeddings = BedrockEmbeddings(model_id=embedding_model_id, client=bedrock)
except Exception as e:
    st.error(f"Error setting up Bedrock embeddings: {e}")
    st.stop()

def data_ingestion():
    try:
        loader = PyPDFDirectoryLoader("data")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        docs = text_splitter.split_documents(documents)
        
        if not docs:
            st.warning("No valid text documents found after ingestion and splitting.")
            st.stop()
        
        return docs
    except Exception as e:
        st.error(f"Error during data ingestion: {e}")
        st.stop()

def get_vector_store(docs):
    try:
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        vectorstore_faiss.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating or saving vector store: {e}")
        st.stop()

def get_claude_llm():
    try:
        llm = Bedrock(model_id=llm_claude_model_id, client=bedrock, model_kwargs={'maxTokenCount': 512})
        return llm
    except Exception as e:
        st.error(f"Error creating Claude LLM: {e}")
        st.stop()

def get_llama2_llm():
    try:
        llm = Bedrock(model_id=llm_llama2_model_id, client=bedrock, model_kwargs={'max_gen_len': 512})
        return llm
    except Exception as e:
        st.error(f"Error creating Llama2 LLM: {e}")
        st.stop()

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        answer = qa({"query": query})
        return answer['result']
    except Exception as e:
        st.error(f"Error getting response from LLM: {e}")
        st.stop()

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            try:
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_claude_llm()
                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")
            except Exception as e:
                st.error(f"Error processing Claude output: {e}")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            try:
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_llama2_llm()
                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")
            except Exception as e:
                st.error(f"Error processing Llama2 output: {e}")

if __name__ == "__main__":
    main()
