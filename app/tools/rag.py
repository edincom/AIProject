from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.config.settings import EMBED_MODEL

# Default splitter for main content
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Splitter for themes/chapters (larger chunks)
theme_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

embeddings = MistralAIEmbeddings(model=EMBED_MODEL)

def format_docs(docs):
    """Format retrieved documents with page numbers"""
    return "\n\n".join(f"[Page {d.metadata.get('page','N/A')}] {d.page_content}" for d in docs)

def split_docs(docs, chunk_size=None, chunk_overlap=None):
    """
    Split documents into smaller chunks for better retrieval.
    Uses the configured splitter (chunk_size, chunk_overlap) to break down
    large documents into manageable pieces that fit within LLM context windows.
    
    Args:
        docs: List of documents to split
        chunk_size: Optional custom chunk size (uses default if None)
        chunk_overlap: Optional custom chunk overlap (uses default if None)
    
    Returns:
        List of split document chunks
    """
    if chunk_size is not None or chunk_overlap is not None:
        # Create custom splitter with specified parameters
        custom_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or CHUNK_SIZE,
            chunk_overlap=chunk_overlap or CHUNK_OVERLAP
        )
        return custom_splitter.split_documents(docs)
    
    # Use default splitter
    return splitter.split_documents(docs)

def build_vectorstore(documents, faiss_path):
    """
    Create a new FAISS vector database from documents.
    - Converts each document chunk into a vector embedding using Mistral's embedding model
    - Stores these embeddings in a FAISS index for fast similarity search
    - Saves the index to disk so it can be reused without rebuilding
    - Returns the vectorstore object for immediate use
    
    Args:
        documents: List of document chunks to embed
        faiss_path: Path where the FAISS index should be saved
    
    Returns:
        FAISS vectorstore object
    """
    store = FAISS.from_documents(documents, embeddings)
    store.save_local(faiss_path)
    return store

def load_vectorstore(faiss_path):
    """
    Load an existing FAISS vector database from disk.
    - Reads the previously saved FAISS index from the specified path
    - Reconstructs the vectorstore with the same embeddings model
    - allow_dangerous_deserialization=True is needed because FAISS uses pickle
    - Much faster than rebuilding the entire index from scratch
    
    Args:
        faiss_path: Path to the saved FAISS index
    
    Returns:
        FAISS vectorstore object
    """
    return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)


def get_retriever(all_docs, faiss_path, search_k=5):
    """
    Get a retriever for semantic search over documents.
    - First attempts to load a pre-built FAISS index from disk (fast)
    - If no index exists, builds a new one from all_docs (slow, first-time only)
    - Returns a retriever object that can find relevant document chunks
      based on semantic similarity to a query
    - This retriever is used by RAG to find context for answering questions
    
    Args:
        all_docs: List of document chunks (only used if building new index)
        faiss_path: Path to save/load the FAISS index
        search_k: Number of similar documents to retrieve (default: 5)
    
    Returns:
        FAISS retriever object configured with search_k
    """
    try:
        store = load_vectorstore(faiss_path)
        print(f"Loaded prebuilt FAISS index from {faiss_path}")
    except:
        store = build_vectorstore(all_docs, faiss_path)
        print(f"Built new FAISS index at {faiss_path}")

    return store.as_retriever(search_kwargs={"k": search_k})