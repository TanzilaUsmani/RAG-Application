import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="📚 RAG Book Assistant",
    page_icon="📖",
    layout="wide"
)

st.title("📚 AI Book Question Answering System")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

embedding_model = HuggingFaceEmbeddings()

llm = ChatMistralAI(
    model="mistral-small-2503"
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI Assistant.

Answer ONLY from the provided context.

If the answer is not found in the document say:

"I could not find the answer in the uploaded document."
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)

if uploaded_file:

    if os.path.exists("uploads") is False:
        os.makedirs("uploads")

    pdf_path = os.path.join("uploads", uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Reading PDF..."):

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(docs)

    # if os.path.exists("chroma_db"):
    #     shutil.rmtree("chroma_db")

    with st.spinner("Creating Embeddings..."):

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )

    st.success("✅ PDF Processed Successfully")

    question = st.text_input("Ask your question")

    if st.button("Get Answer"):

        with st.spinner("Searching..."):

            docs = retriever.invoke(question)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt.invoke(
                {
                    "context":context,
                    "question":question
                }
            )

            response = llm.invoke(final_prompt)

        st.subheader("Answer")

        st.write(response.content)

        with st.expander("Retrieved Chunks"):

            for i, doc in enumerate(docs):

                st.markdown(f"### Chunk {i+1}")

                st.write(doc.page_content)