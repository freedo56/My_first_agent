import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

load_dotenv()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

st.title("📄 RAG — ถามตอบจาก PDF")

uploaded_file = st.file_uploader("อัพโหลด PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(chunks, embeddings)
    st.success(f"โหลดสำเร็จ {len(chunks)} chunks ครับ")

    question = st.text_input("ถามคำถามจากเอกสาร")

    if question:
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        messages = [
            SystemMessage(content=f"ตอบคำถามจากเอกสารนี้เท่านั้น:\n\n{context}"),
            HumanMessage(content=question)
        ]

        response = llm.invoke(messages)
        st.write(response.content)