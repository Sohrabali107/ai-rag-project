import os
import streamlit as st
from dotenv import load_dotenv

import groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="Policy-GPT", page_icon="📄")
st.title("📄 Policy-GPT: AI Document Assistant")
st.write("Upload a readable digital PDF to ask questions using AI.")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Please add your GROQ_API_KEY in the .env file.")
    st.stop()

# Helper: Automatically discover the best available model for your Groq account
# Helper: Select only true chat completion models
@st.cache_resource
def get_working_groq_model():
  client = groq.Groq(api_key=api_key)
  try:
    available_models = [m.id for m in client.models.list().data]

    # Explicit list of valid chat/completion models on Groq
    chat_candidates = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "gemma2-9b-it",
    ]

    for candidate in chat_candidates:
      if candidate in available_models:
        return candidate

    # Fallback: find any model with chat keywords, excluding guards/whisper/vision
    for m in available_models:
      if any(k in m.lower() for k in ["llama", "gemma", "mixtral", "qwen"]):
        if not any(
            bad in m.lower()
            for bad in ["guard", "whisper", "vision", "embed", "safeguard"]
        ):
          return m

    return "llama-3.1-8b-instant"
  except Exception:
    return "llama-3.1-8b-instant"

active_model = get_working_groq_model()

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Reading and indexing document..."):
        # 1. Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Check if PDF contains actual readable text
        total_text = "".join([doc.page_content.strip() for doc in docs])
        if not total_text:
            st.error("⚠️ This PDF contains scanned images or non-selectable text without OCR. Please upload a standard digital PDF (like an ebook, report, resume, or syllabus).")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            st.stop()

        # 2. Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 3. Local embedding model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 4. Helper formatting
        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        # 5. Connect to the active Groq model
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=active_model,
            temperature=0,
        )

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Use the following context to answer the question.
If the answer cannot be found in the context, say that you don't know based on the document.

Context:
{context}

Question: {question}

Answer:"""
        )

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    st.success(f"Document processed successfully using `{active_model}`!")

    user_query = st.text_input("Ask a question about your document:")
    if user_query:
        with st.spinner("Generating answer..."):
            try:
                answer = rag_chain.invoke(user_query)
                st.write("### Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error querying model: {e}")

    if os.path.exists(temp_path):
        os.remove(temp_path)