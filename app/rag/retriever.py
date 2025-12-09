from .vectorstore import build_vectorstore, load_vectorstore

def get_retriever(all_docs):
    try:
        store = load_vectorstore()
        print("Loaded prebuilt FAISS index.")
    except:
        store = build_vectorstore(all_docs)
        print("Built new FAISS index.")

    return store.as_retriever()
