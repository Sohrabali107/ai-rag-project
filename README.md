# Policy-GPT: AI Document Assistant (RAG)

A Retrieval-Augmented Generation (RAG) application built with LangChain, ChromaDB, and OpenAI to question and analyze PDF documents.

## How It Works

1. **PDF Processing:** Documents are ingested and broken into semantic chunks with overlap using `RecursiveCharacterTextSplitter`.
2. **Vector Storage:** Chunks are vectorized using OpenAI embeddings and indexed in `ChromaDB`.
3. **Contextual Retrieval:** Performs semantic similarity search against the user prompt to supply relevant context to `gpt-4o-mini`.
4. **Interface:** Interactive web UI built using `Streamlit`.

## Setup & Installation
