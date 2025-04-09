import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from typing import List
from groq import Groq

# === Streamlit UI Setup ===
st.set_page_config(page_title="PDF Chat (Groq + LLaMA 4)", layout="wide")
st.title("📄 Chat with your PDF using LLaMA 4 Maverick (Groq)")

# === Embedding Wrapper Compatible with LangChain ===
class SentenceEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = len(self.model.encode("test", convert_to_tensor=False))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_tensor=False).tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

embedding_function = SentenceEmbeddingFunction()

# === PDF Utilities ===
def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

def create_vector_store(chunks):
    vector_store = FAISS.from_documents(chunks, embedding=embedding_function)
    vector_store.save_local("faiss_index")

def load_vector_store():
    return FAISS.load_local("faiss_index", embeddings=embedding_function, allow_dangerous_deserialization=True)

# === Groq Chat Completion ===
def get_llama4_answer(query, context):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables!")
        return "Please set your Groq API key."

    client = Groq(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": "You're a helpful assistant that answers based only on the provided context.",
        },
        {
            "role": "user",
            "content": f"Answer the following based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
        }
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.7,
            max_tokens=1000,
        )
        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"❌ Error calling Groq API: {e}")
        return "Failed to get response from Groq API."

# === Query Handler ===
def answer_user_query(query):
    db = load_vector_store()
    docs = db.similarity_search(query, k=3)

    if not docs:
        st.error("❌ No relevant content found in the PDFs.")
        return

    context = "\n\n".join([doc.page_content for doc in docs])

    with st.spinner("💬 Generating answer from LLaMA 4..."):
        answer = get_llama4_answer(query, context)
        st.markdown("### 🧠 Answer from LLaMA 4:")
        st.success(answer)

# === Main App ===
def main():
    query = st.text_input("💬 Ask a question about your PDFs:")

    if query:
        if os.path.exists("faiss_index/index.faiss"):
            answer_user_query(query)
        else:
            st.warning("⚠️ Please upload and process PDFs first!")

    with st.sidebar:
        st.header("📂 Upload PDFs")
        pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
        if st.button("✅ Submit & Process"):
            if pdf_files:
                with st.spinner("Processing PDFs..."):
                    raw_text = extract_text_from_pdfs(pdf_files)
                    chunks = split_text_into_chunks(raw_text)
                    create_vector_store(chunks)
                    st.success("✅ PDFs processed and indexed successfully!")
            else:
                st.warning("⚠️ Please upload at least one PDF.")

if __name__ == "__main__":
    main()
