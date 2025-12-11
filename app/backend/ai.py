# backend/ai.py
import json
from dotenv import load_dotenv

from app.tools.database import save_result
load_dotenv()

from app.tools.rag import get_retriever
from app.tools.loaders import load_pdf
from app.tools.loaders import caption_images
from app.tools.rag import split_docs
from app.chains.persona_chain import streaming_persona_chain, persona_prompt
from app.chains.test_chain import generate_question_chain, test_chain
from app.config.settings import FAISS_PATH
from app.chains.theme_chain import theme_llm
from app.tools.database import save_teach_interaction


print("Initializing AI backend…")

docs = load_pdf()
image_docs = caption_images()
chunks = split_docs(docs)
all_docs = chunks  # + image_docs if you want to include images
retriever = get_retriever(all_docs, FAISS_PATH)
print("Information retriever initialized")


def get_relevant_chapter(question):
    """Find which chapter is most relevant to the question"""
    try:
        with open("document_analysis.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Find JSON content using markers
        start = content.find("```json")
        if start != -1:
            start = content.find("\n", start) + 1
            end = content.find("```", start)
            structure_text = content[start:end].strip()
        else:
            # Fallback: extract between headers
            structure_section = content.split("DOCUMENT STRUCTURE")[1].split("MAIN THEMES")[0]
            structure_text = structure_section.replace("=" * 60, "").strip()
        
        structure = json.loads(structure_text)
        
        # Ask LLM to match question to chapter
        from langchain_core.prompts import ChatPromptTemplate
        
        chapter_prompt = ChatPromptTemplate.from_messages([
            ("system", "Tu es un assistant qui identifie à quel chapitre correspond une question."),
            ("human", """Question: {question}

            Chapitres: {chapters}

            Réponds avec: "Chapitre X: Titre" ou "Chapitre général".""")
        ])
        
        chain = chapter_prompt | theme_llm
        result = chain.invoke({
            "question": question,
            "chapters": json.dumps(structure.get("chapters", []), indent=2, ensure_ascii=False)
        })
        
        return result.content.strip()
        
    except Exception as e:
        print(f"Error identifying chapter: {e}")
        import traceback
        traceback.print_exc()
        return "Chapitre général"

def ai_answer_stream(inputs, username="Guest", chapter=None):
    """
    Stream answer from the RAG/chat system token by token.
    Saves the interaction to the database after streaming completes.
    
    Args:
        inputs: dict with 'question' key containing the user's question
        username: str - The student's username (default: "Guest")
        chapter: str - The chapter context (optional)
    
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
    
    # Variables to collect data for database
    chapter_context = chapter or ""
    full_answer = ""
    
    try:
        # Step 0: Find the relevant chapter if not provided
        if not chapter_context:
            chapter_context = get_relevant_chapter(question)

        # Step 1: Retrieve context using RAG (non-streaming)
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        
        # Step 2: Get conversation history for this chapter
        from app.tools.database import get_student_chapter_interactions
        history = get_student_chapter_interactions(username, chapter_context)
        history_text = "\n".join([f"Q: {h[1]}\nA: {h[2]}" for h in history[-5:]])  # Last 5 interactions
        
        # Step 3: Stream the LLM response with context
        stream_inputs = {
            "question": question,
            "chapter_context": chapter_context,
            "context": context + (f"\n\nConversation history:\n{history_text}" if history_text else "")
        }

        # Debug: Print what's being sent to the LLM
        print("\n" + "="*80)
        print("🔍 SENDING TO MISTRAL API:")
        print("="*80)
        print(f"Question: {question}")
        print(f"Chapter: {chapter_context}")
        print(f"Context length: {len(context)} chars")
        print(f"History included: {bool(history_text)}")
        print("\nFull prompt inputs:")
        print(stream_inputs)
        print("="*80 + "\n")

        # # Debug: Print raw prompt being ssent
        # print("\n" + "="*80)
        # print("🔍 RAW PROMPT TO MISTRAL:")
        # print("="*80)
        # print(persona_prompt.format(**stream_inputs))
        # print("="*80 + "\n")
        
        # Stream directly from the persona chain
        for chunk in streaming_persona_chain.stream(stream_inputs):
            # Extract only the text content from each chunk
            if hasattr(chunk, 'content'):
                content = chunk.content
                if content:  # Only yield non-empty content
                    full_answer += content
                    yield content
            elif isinstance(chunk, str):
                if chunk:  # Only yield non-empty strings
                    full_answer += chunk
                    yield chunk
        
        # Step 4: Save interaction to database after streaming completes
        if full_answer:  # Only save if we got a response
            save_teach_interaction(username, chapter_context, question, full_answer)
                    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Error in ai_answer_stream: {e}")
        import traceback
        traceback.print_exc()
        yield error_msg
        
        # Save error interaction to database
        if chapter_context or question:
            save_teach_interaction(username, chapter_context or "Error", question, error_msg)


def generate_test_question(criteria):
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
            "- Faits non corrects : Y a-t-il des faits incorrects /30;\n"
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