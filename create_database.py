#load pdf
#splite into chunk
#create embedding
#store  into chroma db

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

data =PyPDFLoader("Documentloader/Deep learning.pdf")
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10

)

chunk=splitter.split_documents(docs)

embedding_model=HuggingFaceEmbeddings()
vectorstore=Chroma.from_documents(
    documents=chunk,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
