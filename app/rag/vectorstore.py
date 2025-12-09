from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.config.settings import EMBED_MODEL, FAISS_PATH

embeddings = MistralAIEmbeddings(model=EMBED_MODEL)

def build_vectorstore(documents):
    store = FAISS.from_documents(documents, embeddings)
    store.save_local(FAISS_PATH)
    return store

def load_vectorstore():
    return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
