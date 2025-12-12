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
from langchain_core.prompts import ChatPromptTemplate
from app.tools.database import get_student_chapter_interactions

print("Initializing AI backend…")

docs = load_pdf()
image_docs = caption_images()
chunks = split_docs(docs)
all_docs = chunks  # + image_docs if you want to include images
retriever = get_retriever(all_docs, FAISS_PATH)
print("Information retriever initialized")


def get_relevant_chapter(question, recent_history=None):
    """Find which chapter is most relevant to the question, considering conversation context"""
    try:
        with open("document_analysis.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        start = content.find("```json")
        if start != -1:
            start = content.find("\n", start) + 1
            end = content.find("```", start)
            structure_text = content[start:end].strip()
        else:
            structure_section = content.split("DOCUMENT STRUCTURE")[1].split("MAIN THEMES")[0]
            structure_text = structure_section.replace("=" * 60, "").strip()
        
        structure = json.loads(structure_text)
        
        # Build conversation context if available
        conversation_context = ""
        previous_chapter = "Aucun"
        if recent_history and len(recent_history) > 0:
            previous_chapter = recent_history[0][1]  # Get the chapter from most recent interaction
            conversation_context = "\n\nConversation récente:\n" + "\n".join([
                f"Chapitre: {h[1]}\nQ: {h[2]}\nR: {h[3][:150]}..." for h in recent_history[:2]
            ])
        
        chapter_prompt = ChatPromptTemplate.from_messages([
            ("system", """
Tu es un classifieur avancé chargé de deux tâches :
1) Déterminer si la question actuelle est une SUITE (suite logique) de la conversation.
2) Identifier le CHAPITRE approprié.

-----------------------------------
RÈGLE 1 — DÉTECTION DE SUIVI (OUI/NON)
-----------------------------------
Réponds SUIVI: OUI si et seulement si la question dépend du contexte précédent pour être comprise.

Réponds SUIVI: NON si la question est autonome, introduit un nouveau sujet ou peut être comprise sans le contexte.

En cas de doute → NON.

INDICATEURS FORTS DE SUIVI (OUI) :
- Pronoms ou références sans antécédent explicite : il, elle, ça, cela, celui-ci, celui-là…
- Questions elliptiques / fragmentaires : « Et après ? », « Et lui ? », « Pourquoi ça ? »…
- Demandes de clarification : « Que veux-tu dire par X ? »
- Référence à un élément uniquement présent dans l'historique : « la deuxième », « cette partie », « cette date »
- Poursuite naturelle du même sujet introduit précédemment.

INDICATEURS DE NON-SUIVI (NON) :
- Thème totalement différent.
- Question autonome sans dépendance au contexte.
- Changement de sujet explicite ou implicite.
- Reformulation vague sans lien précis.

-----------------------------------
RÈGLE 2 — ATTRIBUTION DE CHAPITRE
-----------------------------------
- Si SUIVI: OUI → CHAPITRE: Reprends le chapitre de la conversation récente (fourni dans le contexte)
- Si SUIVI: NON → analyse la question seule et sélectionne le chapitre le plus pertinent des chapitres disponibles.
- Si aucun chapitre ne correspond → « Chapitre général ».

Chapitre précédent: {previous_chapter}

-----------------------------------
FORMAT DE SORTIE — STRICT, 3 LIGNES
-----------------------------------
SUIVI: OUI ou NON
RAISON: phrase brève (10–20 mots) indiquant l'indicateur utilisé
CHAPITRE: Chapitre X: Titre OU Chapitre général (écris le nom complet du chapitre, PAS "Même chapitre")
-----------------------------------
Aucune autre ligne. Aucun ajout, justification longue ou commentaire.
-----------------------------------

EXEMPLES INTERNES (NE PAS REPRODUIRE DANS LA SORTIE) :

1. Conversation : « Napoléon a été exilé… » (Chapitre 5) — Question : « Et après, qu'a-t-il fait ? »
→ SUIVI: OUI (pronom référentiel), CHAPITRE: Chapitre 5: [Titre exact]

2. Conversation : « La 1ʳᵉ GM commence en 1914 » (Chapitre 3) — Question : « Et ça dure combien de temps ? »
→ SUIVI: OUI (référence contextuelle), CHAPITRE: Chapitre 3: [Titre exact]

3. Conversation : « L'Europe compte 27 États » (Chapitre 4) — Question : « Comment fonctionne la démocratie athénienne ? »
→ SUIVI: NON (changement de sujet), CHAPITRE: Chapitre 5: Retour sur l'histoire

"""),

            ("human", """
Conversation récente :
{conversation_context}

Question actuelle :
{question}

Chapitres disponibles :
{chapters}

Analyse et réponds strictement au format demandé.
""")
        ])

        
        # DEBUG: Print what we're sending to the LLM
        print("\n" + "="*80)
        print("🔍 CHAPTER DETECTION - INPUT:")
        print("="*80)
        print(f"Question: {question}")
        print(f"Previous chapter: {previous_chapter}")
        print(f"Has recent history: {bool(recent_history)}")
        if recent_history:
            print(f"Recent history count: {len(recent_history)}")
            print("Recent conversation context:")
            print(conversation_context)
        print("="*80 + "\n")
        
        chain = chapter_prompt | theme_llm
        result = chain.invoke({
            "question": question,
            "conversation_context": conversation_context,
            "previous_chapter": previous_chapter,
            "chapters": json.dumps(structure.get("chapters", []), indent=2, ensure_ascii=False)
        })
        
        response = result.content.strip()
        
        # Parse the response
        lines = response.split('\n')
        detected_chapter = "Chapitre général"
        reason = "Aucune raison fournie"
        is_followup = "NON"
        
        for line in lines:
            if line.startswith("SUIVI:"):
                is_followup = line.replace("SUIVI:", "").strip()
            elif line.startswith("CHAPITRE:"):
                detected_chapter = line.replace("CHAPITRE:", "").strip()
            elif line.startswith("RAISON:"):
                reason = line.replace("RAISON:", "").strip()
        
        # DEBUG: Print what the LLM decided
        print("\n" + "="*80)
        print("✅ CHAPTER DETECTION - OUTPUT:")
        print("="*80)
        print(f"Is follow-up: {is_followup}")
        print(f"Detected chapter: {detected_chapter}")
        print(f"Reason: {reason}")
        print("="*80 + "\n")
        
        return detected_chapter
        
    except Exception as e:
        print(f"❌ Error identifying chapter: {e}")
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
    # Import at the beginning
    from app.tools.database import get_student_chapter_interactions
    
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
            # Get recent conversation for context
            all_recent = get_student_chapter_interactions(username, None)
            print(f"🔍 DEBUG: Fetched {len(all_recent)} recent interactions for user '{username}'")
            if all_recent:
                print(f"   Most recent: {all_recent[0]}")
            
            # Pass recent history to chapter identifier
            chapter_context = get_relevant_chapter(question, recent_history=all_recent[:3])

        # Step 1: Retrieve context using RAG (non-streaming)
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        
        # Step 2: Get conversation history for this chapter
        history = get_student_chapter_interactions(username, chapter_context)
        history_text = "\n".join([f"Q: {h[2]}\nA: {h[3]}" for h in history[-5:]])  # Last 5 interactions
        
        # Step 3: Stream the LLM response with context
        stream_inputs = {
            "question": question,
            "chapter_context": chapter_context,
            "context": context,
            "history": history_text if history_text else "Aucune conversation précédente"
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
        
        # Token counting variables
        token_count_input = 0
        token_count_output = 0
        
        # Stream directly from the persona chain
        for chunk in streaming_persona_chain.stream(stream_inputs):
            # Extract only the text content from each chunk
            if hasattr(chunk, 'content'):
                content = chunk.content
                if content:  # Only yield non-empty content
                    full_answer += content
                    token_count_output += len(content.split())  # Rough estimate
                    yield content
            elif isinstance(chunk, str):
                if chunk:  # Only yield non-empty strings
                    full_answer += chunk
                    token_count_output += len(chunk.split())
                    yield chunk
            
            # Check if chunk has usage metadata (safe check)
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata is not None:
                token_count_input = chunk.usage_metadata.get('input_tokens', 0)
        
        # Print token usage after streaming completes
        if token_count_input > 0:
            print(f"\n📊 Token usage - Input: {token_count_input}, Output (estimated): {token_count_output}")
        else:
            print(f"\n📊 Token usage - Input: Not available from API, Output (estimated): {token_count_output}")
        
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