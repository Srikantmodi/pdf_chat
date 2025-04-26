import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
import faiss

load_dotenv()  # Important! This will read your .env

groq_api_key = os.getenv("GROQ_API_KEY")

print(f"Groq API Key: {groq_api_key}")  # Check if it loads

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)

# Function to create a vectorstore
def get_vectorstore(text_chunks, embeddings):
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Main app
def main():
    load_dotenv()  # Load environment variables
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
    st.header("Chat with your PDFs 📚")

    # Initialize session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)

        st.subheader("Choose Embedding Model")
        embedding_choice = st.selectbox(
            "Select the embedding model",
            ["HuggingFace - all-MiniLM-L6-v2 (No token needed)", "Instructor - hkunlp/instructor-xl (Needs torch/transformers)"]
        )

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # 1. Read PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    # 2. Split into chunks
                    text_chunks = get_text_chunks(raw_text)
                    # 3. Load Embeddings
                    if embedding_choice.startswith("HuggingFace"):
                        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    else:
                        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
                    # 4. Create vectorstore and save in session_state
                    st.session_state.vectorstore = get_vectorstore(text_chunks, embeddings)
                st.success("PDFs processed and vectorstore created!")

    # Input for user's question
    query = st.text_input("Ask a question about your PDFs:")
    if query:
        if st.session_state.vectorstore is not None:
            with st.spinner("Searching for answers..."):
                docs = st.session_state.vectorstore.similarity_search(query, k=3)

                # ✅ Using Groq as LLM for final answer generation
                llm = ChatGroq(
                    temperature=0,
                    model_name="Llama3-8b-8192",  # You can use "Llama3-70b-8192" if you want bigger model
                    groq_api_key=groq_api_key,
                )

                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=query)

                st.subheader("Answer:")
                st.write(response)
        else:
            st.warning("Please upload and process PDFs first!")

if __name__ == "__main__":
    main()
