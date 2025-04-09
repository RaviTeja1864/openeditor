import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# === Load environment variables ===
load_dotenv(dotenv_path=".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not GOOGLE_API_KEY or not GOOGLE_CREDENTIALS:
    raise EnvironmentError("❌ Please check your .env file. GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS is missing.")

# Configure API access
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS
genai.configure(api_key=GOOGLE_API_KEY)

# === PDF Utilities ===
def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def build_qa_chain():
    prompt_template = """
    Answer the question based on the provided context.
    If the answer is not available, say "The answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="models/chat-bison-001", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def answer_user_query(query):
    db = load_vector_store()
    relevant_docs = db.similarity_search(query)
    chain = build_qa_chain()
    result = chain({"input_documents": relevant_docs, "question": query}, return_only_outputs=True)
    st.write("📌 **Answer:**", result["output_text"])

# === Streamlit UI ===
def main():
    st.set_page_config(page_title="Chat with your PDF")
    st.header("📄 Chat with your PDF using Google AI")

    query = st.text_input("💬 Ask a question about your PDFs:")

    if query:
        answer_user_query(query)

    with st.sidebar:
        st.title("📂 Upload your PDFs")
        pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("✅ Submit & Process"):
            if pdf_files:
                with st.spinner("Processing PDFs..."):
                    text = extract_text_from_pdfs(pdf_files)
                    chunks = split_text_into_chunks(text)
                    create_vector_store(chunks)
                    st.success("✅ PDFs processed and indexed!")
            else:
                st.warning("⚠️ Please upload at least one PDF.")

if __name__ == "__main__":
    main()
