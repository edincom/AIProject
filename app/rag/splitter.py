from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config.settings import CHUNK_SIZE, CHUNK_OVERLAP

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def split_docs(docs):
    return splitter.split_documents(docs)
