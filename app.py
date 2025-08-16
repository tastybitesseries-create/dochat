import streamlit as st
import tempfile
import os

from utils.pdf_loader import extract_text
from utils.embedder import embed_texts, query_llm

st.set_page_config(page_title="Chat with PDF MVP", page_icon="📄")

st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Extract text
    text = extract_text(pdf_path)

    # Embed + store
    vectorstore = embed_texts(text)

    st.success("PDF processed! Ask your question below 👇")

    query = st.text_input("Your Question:")
    if query:
        answer = query_llm(query, vectorstore)
        st.write("**Answer:**", answer)
