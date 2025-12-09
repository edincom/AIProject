from langchain_community.document_loaders import PyMuPDFLoader
from app.config.settings import PDF_PATH

def load_pdf():
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")
    return docs
