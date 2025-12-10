# backend/ai.py
import json
from dotenv import load_dotenv

from app.tools.database import save_result
load_dotenv()

from app.tools.rag import get_retriever
from app.tools.loaders import load_pdf
from app.tools.loaders import caption_images
from app.tools.rag import split_docs
from app.chains.persona_chain import streaming_persona_chain
from app.chains.test_chain import generate_question_chain, test_chain
from app.chains.theme_chain import theme_llm
from app.chains.theme_chain import extract_chapters_and_themes

print("Initializing AI backend…")

docs = load_pdf()

#Theme of the content

structure, themes, retriever_theme = extract_chapters_and_themes(docs, theme_llm)

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


def generate_test_question(criteria, historic):
    """
    Generate a test question using document context (RAG) and student criteria.

    The student can choose whether the question is completely random (within the given subject), or the question is based
    on the questions he poorly answered previously, as all the questions are stored in a database.
    
    Args:
        criteria: str - The topic/criteria the student wants to be tested on
    
    Returns:
        str: Generated question
    """

    try:
        # Step 1: Retrieve relevant context using RAG
        docs = retriever.invoke(criteria)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # Step 2: Generate question using the context
        result = generate_question_chain.invoke({
            "criteria": criteria,
            "context": context_text
        })
        
        # Extract the question from the result
        if hasattr(result, 'content'):
            question = result.content.strip()
        else:
            question = str(result).strip()
        
        return question
        
    except Exception as e:
        print(f"Error in generate_test_question: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating question: {str(e)}"
    
def grade_answer(question, answer, rubric, username="Anonymous"):
    """
    Grade a student's answer and save to database.
    
    Args:
        question: str - The question that was asked
        answer: str - The student's answer
        rubric: str - The grading rubric/criteria
        username: str - The student's username
    
    Returns:
        dict: Grading result with grade, scores, and advice
    """
    try:
        # Define the scoring template
        scores_text = (
            "- Pertinence : Est-ce que l'étudiant répond bien à la question posée /30;\n"
            "- Faits non correctes : Y a-t-il des faits incorrects /30;\n"
            "- Faits manquants : Tous les faits attendus sont-ils présents /30;\n"
            "- Structure : La réponse est-elle bien structurée /10;"
        )
        
        # Invoke the test chain
        result = test_chain.invoke({
            "grading_rubric": rubric,
            "question": question,
            "answer": answer,
            "scores_text": scores_text
        })
        
        # Extract content from result
        if hasattr(result, 'content'):
            raw_output = result.content.strip()
        else:
            raw_output = str(result).strip()
        
        # Clean up markdown formatting if present
        if raw_output.startswith("```"):
            lines = raw_output.split("\n")
            # Remove first and last lines if they're markdown fences
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_output = "\n".join(lines)
        
        # Parse JSON
        try:
            grading_json = json.loads(raw_output)
        except json.JSONDecodeError as je:
            print(f"JSON decode error: {je}")
            print(f"Raw output: {raw_output}")
            # Return a default structure
            grading_json = {
                "Section": "Unknown",
                "Question": question,
                "Answer": answer,
                "grade": 0,
                "scores": {
                    "Pertinence": 0,
                    "Faits non correctes": 0,
                    "Faits manquants": 0,
                    "Structure": 0
                },
                "advice": "Error parsing grading result. Please try again."
            }
        
        # Save to database
        try:
            save_result(username, question, answer, grading_json)
            print(f"Result saved for {username}")
        except Exception as db_error:
            print(f"Database error: {db_error}")
            # Don't fail the grading if database save fails
        
        return grading_json
        
    except Exception as e:
        print(f"Error in grade_answer: {e}")
        import traceback
        traceback.print_exc()
        return {
            "Section": "Error",
            "Question": question,
            "Answer": answer,
            "grade": 0,
            "scores": {
                "Pertinence": 0,
                "Faits non correctes": 0,
                "Faits manquants": 0,
                "Structure": 0
            },
            "advice": f"Error during grading: {str(e)}"
        }