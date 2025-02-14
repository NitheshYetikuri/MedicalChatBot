# MedicalChatBot

This project is a medical chatbot built using LangChain and Streamlit. It loads and processes medical documents to provide accurate answers to user queries. The chatbot uses FAISS for efficient document retrieval and ChatDeepSeek for generating responses. Users can interact with the chatbot through a user-friendly Streamlit interface.

## Features
- **Document Loading**: Utilizes `PyPDFLoader` to load medical documents.
- **Text Splitting**: Uses `RecursiveCharacterTextSplitter` to split documents into manageable chunks.
- **Vector Store**: Employs `FAISS` with `DeepSeek` embeddings for efficient document retrieval.
- **Language Models**: Integrates `ChatDeepSeek` for generating responses based on the retrieved information.
- **Streamlit Interface**: Provides a user-friendly interface for users to input their medical questions and receive answers.

## How It Works
1. **Load Documents**: Medical documents are loaded and split into chunks.
2. **Create Vector Store**: The chunks are embedded and stored in a vector store for efficient retrieval.
3. **Generate Responses**: A retrieval chain is used to find relevant information and generate responses to user queries.

## Requirements
- `langchain`
- `langchain_community`
- `langchain_ollama`
- `langchain_core`
- `langchain_google_genai`
- `streamlit`
- `faiss`

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/NitheshYetikuri/MedicalChatBot.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
- Run the Streamlit app to interact with the chatbot.
- Ask medical questions and receive concise, document-based answers.

## Models Used
- **Embeddings**: `DeepSeek` model for creating embeddings.
- **LLM**: `ChatDeepSeek` model for generating responses.

## Document Source
The medical information is sourced from "The GALE Encyclopedia of Medicine, Second Edition" PDF provided in the repository.
