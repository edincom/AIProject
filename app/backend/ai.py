# backend/ai.py
from dotenv import load_dotenv
load_dotenv()


from app.chains.rag_router import build_router
from app.rag.retriever import get_retriever
from app.loaders.pdf_loader import load_pdf
from app.loaders.image_captioning import caption_images
from app.rag.splitter import split_docs

# You ONLY build the RAG system once, on import.
print("Initializing AI backend…")

docs = load_pdf()
image_docs = caption_images()
chunks = split_docs(docs)
all_docs = chunks #+ image_docs

retriever = get_retriever(all_docs)
print("to")
router = build_router(retriever)
print("ta")

def ai_answer(question: str):
    """Return answer from the RAG/chat system"""
    return router.invoke({"question": question})
