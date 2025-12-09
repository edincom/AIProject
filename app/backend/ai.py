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
all_docs = chunks  # + image_docs if you want to include images

retriever = get_retriever(all_docs)
print("Retriever initialized")

router = build_router(retriever)
print("Router initialized")

def ai_answer(inputs):
    """
    Return answer from the RAG/chat system
    
    Args:
        inputs: dict with 'question' key containing the user's question
    
    Returns:
        AIMessage with the response
    """
    # Ensure inputs is a dict with 'question' key
    if isinstance(inputs, str):
        inputs = {"question": inputs}
    
    if not isinstance(inputs, dict):
        raise TypeError(f"Expected dict or str, got {type(inputs)}")
    
    if "question" not in inputs:
        raise ValueError("inputs dict must contain 'question' key")
    
    # Ensure question is a string
    question = inputs["question"]
    if not isinstance(question, str):
        question = str(question)
    
    # Create clean input dict
    clean_inputs = {"question": question.strip()}
    
    try:
        return router.invoke(clean_inputs)
    except Exception as e:
        print(f"Error in router.invoke: {e}")
        import traceback
        traceback.print_exc()
        raise