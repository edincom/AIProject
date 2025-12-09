# backend/ai.py
from dotenv import load_dotenv
load_dotenv()

from app.tools.rag import get_retriever
from app.tools.loaders import load_pdf
from app.tools.loaders import caption_images
from app.tools.rag import split_docs
from app.chains.persona_chain import streaming_persona_chain

print("Initializing AI backend…")

docs = load_pdf()
image_docs = caption_images()
chunks = split_docs(docs)
all_docs = chunks  # + image_docs if you want to include images
retriever = get_retriever(all_docs)
print("Retriever initialized")


def ai_answer_stream(inputs):
    """
    Stream answer from the RAG/chat system token by token
    
    Args:
        inputs: dict with 'question' key containing the user's question
    
    Yields:
        str: Individual tokens/chunks of the response
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
    
    question = question.strip()
    
    try:
        # Step 1: Retrieve context using RAG (non-streaming)
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        
        # Step 2: Stream the LLM response with context
        stream_inputs = {
            "question": question,
            "context": context
        }
        
        # Stream directly from the persona chain
        for chunk in streaming_persona_chain.stream(stream_inputs):
            # Extract only the text content from each chunk
            if hasattr(chunk, 'content'):
                content = chunk.content
                if content:  # Only yield non-empty content
                    yield content
            elif isinstance(chunk, str):
                if chunk:  # Only yield non-empty strings
                    yield chunk
                    
    except Exception as e:
        print(f"Error in ai_answer_stream: {e}")
        import traceback
        traceback.print_exc()
        yield f"Error: {str(e)}"