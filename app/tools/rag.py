from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.config.settings import EMBED_MODEL, FAISS_PATH

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def split_docs(docs):
    return splitter.split_documents(docs)

embeddings = MistralAIEmbeddings(model=EMBED_MODEL)

def build_vectorstore(documents):
    store = FAISS.from_documents(documents, embeddings)
    store.save_local(FAISS_PATH)
    return store

def load_vectorstore():
    return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)



def get_retriever(all_docs):
    try:
        store = load_vectorstore()
        print("Loaded prebuilt FAISS index.")
    except:
        store = build_vectorstore(all_docs)
        print("Built new FAISS index.")

    return store.as_retriever()







