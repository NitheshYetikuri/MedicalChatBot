from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

def get_docs():
    loader = PyPDFLoader(r"Provide document link here of medicine information")
    data = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(data)
    return docs

def get_vectorstore(docs):
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def get_generation(retriever, query):
    llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0.1)
    system_prompt = (
        """
        You are a medical chatbot. Your task is to answer questions related to medicines based on the provided document. If you cannot find an exact match for a medicine related to a specific disease, advise the user to consult a doctor.

        Instructions:

        1. Only provide answers if there is an exact match for the disease in the document.
        2. If no exact match is found, respond with: "Please consult a doctor for accurate medical advice. You can reach out to Dr. Nithesh Yetikuri Medical ChatBot Clinic, Madanapalle, Andhra Pradesh, India, 517325. Contact: yetikurinithesh@gmail.com."
        3. Keep your answers concise and to the point.

        {context}
        """
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    QChain = create_stuff_documents_chain(llm, prompt)
    Chain = create_retrieval_chain(retriever, QChain)

    return Chain.invoke({"input": query})['answer']

docs = get_docs()
vectorstore = get_vectorstore(docs)
retriever = vectorstore.as_retriever(k=5)

st.title("Medical ChatBot")
st.write("Ask your medical questions below:")

query = st.chat_input("Enter your question:", key="query_input")
if query:
    response = get_generation(retriever, query)
    st.write(response)