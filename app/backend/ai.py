# backend/ai.py
import json
from dotenv import load_dotenv

from app.tools.database import save_result
load_dotenv()

from app.tools.rag import add_question_to_faiss, get_questions_retriever, get_retriever
from app.tools.loaders import load_pdf
from app.tools.loaders import caption_images
from app.tools.rag import split_docs
from app.chains.persona_chain import streaming_persona_chain, persona_prompt
from app.chains.test_chain import generate_question_chain, test_chain
from app.config.settings import FAISS_PATH, FAISS_QUESTIONS_PATH, LLM_MODEL
from app.chains.theme_chain import theme_llm
from app.tools.database import save_teach_interaction
from langchain_core.prompts import ChatPromptTemplate
from app.tools.database import get_student_chapter_interactions
from app.tools.ecologits_tracker import tracker


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
- Si SUIVI: OUI → IGNORE cette règle, le code utilisera automatiquement le chapitre précédent
- Si SUIVI: NON → analyse la question seule et sélectionne le chapitre le plus pertinent des chapitres disponibles.
- Si aucun chapitre ne correspond → « Chapitre général ».

Chapitre précédent: {previous_chapter}

-----------------------------------
FORMAT DE SORTIE — STRICT, 3 LIGNES
-----------------------------------
SUIVI: OUI ou NON
RAISON: phrase brève (10–20 mots) indiquant l'indicateur utilisé
CHAPITRE: Chapitre X: Titre OU Chapitre général (UNIQUEMENT si SUIVI: NON)
-----------------------------------
Aucune autre ligne. Aucun ajout, justification longue ou commentaire.
-----------------------------------

EXEMPLES INTERNES (NE PAS REPRODUIRE DANS LA SORTIE) :

1. Conversation : « Napoléon a été exilé… » (Chapitre 5) — Question : « Et après, qu'a-t-il fait ? »
→ SUIVI: OUI (pronom référentiel), CHAPITRE sera ignoré (code utilise Chapitre 5)

2. Conversation : « La 1ʳᵉ GM commence en 1914 » (Chapitre 3) — Question : « Et ça dure combien de temps ? »
→ SUIVI: OUI (référence contextuelle), CHAPITRE sera ignoré (code utilise Chapitre 3)

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
        
        # CRITICAL FIX: If follow-up detected, use previous_chapter directly
        # This ensures consistency - we don't rely on LLM to echo back the chapter
        if "OUI" in is_followup.upper() and previous_chapter != "Aucun":
            detected_chapter = previous_chapter
            reason = f"Suite logique détectée → même chapitre: {previous_chapter}"
        
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
    Enregistre aussi les impacts environnementaux avec EcoLogits.
    
    Args:
        inputs: dict with 'question' key containing the user's question
        username: str - The student's username (default: "Guest")
        chapter: str - The chapter context (optional)
    
    Yields:
        str: Individual tokens/chunks of the response
    """
    
    if isinstance(inputs, str):
        inputs = {"question": inputs}
    
    if not isinstance(inputs, dict):
        raise TypeError(f"Expected dict or str, got {type(inputs)}")
    
    if "question" not in inputs:
        raise ValueError("inputs dict must contain 'question' key")
    
    question = inputs["question"]
    if not isinstance(question, str):
        question = str(question)
    
    question = question.strip()
    
    chapter_context = chapter or ""
    full_answer = ""
    
    try:
        if not chapter_context:
            all_recent = get_student_chapter_interactions(username, None)
            print(f"🔍 DEBUG: Fetched {len(all_recent)} recent interactions for user '{username}'")
            if all_recent:
                print(f"   Most recent: {all_recent[0]}")
            
            chapter_context = get_relevant_chapter(question, recent_history=all_recent[:3])

        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        
        history = get_student_chapter_interactions(username, chapter_context)
        history_text = "\n".join([f"Q: {h[2]}\nA: {h[3]}" for h in history[-5:]])
        
        stream_inputs = {
            "question": question,
            "chapter_context": chapter_context,
            "context": context,
            "history": history_text if history_text else "Aucune conversation précédente"
        }

        print("\n" + "="*80)
        print("🔍 SENDING TO MISTRAL API:")
        print("="*80)
        print(f"Question: {question}")
        print(f"Chapter: {chapter_context}")
        print(f"Context length: {len(context)} chars")
        print(f"History included: {bool(history_text)}")
        print("="*80 + "\n")
        
        token_count_output = 0
        
        for chunk in streaming_persona_chain.stream(stream_inputs):
            if hasattr(chunk, 'content'):
                content = chunk.content
                if content:
                    full_answer += content
                    token_count_output += len(content.split())
                    yield content
            elif isinstance(chunk, str):
                if chunk:
                    full_answer += chunk
                    token_count_output += len(chunk.split())
                    yield chunk
        
        print(f"\n📊 Token usage - Output (estimated): {token_count_output}")
        
        # 🌱 NOUVEAU : Enregistrer les impacts EcoLogits
        llm_instance = streaming_persona_chain.llm
        impacts = llm_instance.get_last_impacts()
        if impacts:
            tracker.record_impact(
                mode="teach",
                username=username,
                energy=impacts["energy"]["value"],
                gwp=impacts["gwp"]["value"],
                model=LLM_MODEL,
                operation="chat"
            )
        
        if full_answer:
            save_teach_interaction(username, chapter_context, question, full_answer)
                    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Error in ai_answer_stream: {e}")
        import traceback
        traceback.print_exc()
        yield error_msg
        
        if chapter_context or question:
            save_teach_interaction(username, chapter_context or "Error", question, error_msg)



def generate_test_question(criteria, allow_reuse=True, similarity_threshold=0.75):
    """
    Generate a test question using document context (RAG).
    Automatically reuses existing questions when possible.
    """
    
    if allow_reuse:
        print(f"\n🔍 Searching similar question (criteria: '{criteria}')")
        
        questions_retriever = get_questions_retriever(
            similarity_threshold=similarity_threshold,
            k=1
        )
        
        if questions_retriever:
            results = questions_retriever.invoke(criteria)
            
            if results:
                doc = results[0]
                print(f"♻️ Question réutilisée!")
                print(f"   💚 Économie estimée : ~1800 tokens ≈ 6g CO2")
                
                return {
                    'question': doc.page_content,
                    'expected_answer': doc.metadata.get('expected_answer', ''),
                    'key_points': doc.metadata.get('key_points', []),
                    'context_used': ''
                }
            else:
                print(f"❌ No match above {similarity_threshold:.0%} threshold")
        else:
            print("ℹ️ Questions FAISS not ready yet (need 5+ questions)")

    try:
        docs = retriever.invoke(criteria)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        result = generate_question_chain.invoke({
            "criteria": criteria,
            "context": context_text
        })

        raw_output = result.content if hasattr(result, "content") else str(result)
        raw_output = raw_output.strip()

        if raw_output.startswith("```"):
            lines = raw_output.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_output = "\n".join(lines).strip()

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            import re
            m = re.search(r'\{[\s\S]*\}', raw_output)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception as e:
                    print("JSON fallback parse failed:", e)
                    parsed = None
            else:
                parsed = None

        if not parsed:
            print("Failed to parse JSON from LLM output. Raw output:")
            print(raw_output)
            return {
                "question": "ERROR_GENERATING_QUESTION",
                "expected_answer": "",
                "key_points": [],
                "context_used": context_text
            }
        
        question = parsed.get("question", "").strip()
        expected_answer = parsed.get("expected_answer", "").strip()
        key_points = parsed.get("key_points") or []
        if isinstance(key_points, str):
            key_points = [kp.strip() for kp in key_points.split("\n") if kp.strip()]
        
        dict_questions = {
            "question": question,
            "expected_answer": expected_answer,
            "key_points": key_points,
            "context_used": context_text
        }

        # 🌱 NOUVEAU : Enregistrer les impacts EcoLogits
        llm_instance = generate_question_chain.llm
        impacts = llm_instance.get_last_impacts()
        if impacts:
            tracker.record_impact(
                mode="test",
                username="system",  # Pas d'utilisateur spécifique lors de la génération
                energy=impacts["energy"]["value"],
                gwp=impacts["gwp"]["value"],
                model=LLM_MODEL,
                operation="generate_question"
            )

        if question and question not in ["ERROR_GENERATING_QUESTION", "ERROR"]:
            try:
                print("Adding to questions FAISS...")
                add_question_to_faiss(dict_questions, FAISS_QUESTIONS_PATH)
            except Exception as e:
                print(f"⚠️ Failed to add to FAISS: {e}")
        
        print("Generated question:", dict_questions)
        return dict_questions

    except Exception as e:
        print(f"Error in generate_test_question: {e}")
        import traceback
        traceback.print_exc()
        return {
            "question": "ERROR",
            "expected_answer": "",
            "key_points": [],
            "context_used": ""
        }


    
def grade_answer(question, answer, expected_answer, key_points, username="Anonymous"):
    """
    Grade a student's answer based on objective criteria.
    Enregistre aussi les impacts environnementaux.
    """
    try:
        result = test_chain.invoke({
            "question": question,
            "answer": answer,
            "expected_answer": expected_answer,
            "key_points": key_points
        })
        
        raw_output = result.content if hasattr(result, "content") else str(result)
        raw_output = raw_output.strip()
        
        if raw_output.startswith("```"):
            lines = raw_output.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_output = "\n".join(lines).strip()

        try:
            grading_json = json.loads(raw_output)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[\s\S]*\}', raw_output)
            if match:
                grading_json = json.loads(match.group(0))
            else:
                print("Failed JSON:", raw_output)
                return {
                    "Section": "Error",
                    "Question": question,
                    "Answer": answer,
                    "expected_answer": expected_answer,
                    "key_points": key_points,
                    "grade": 0,
                    "scores": {
                        "Key Points": 0,
                        "Expected Match": 0,
                        "Incorrect Facts": 0,
                        "Structure": 0
                    },
                    "advice": "Parsing error. Try again."
                }

        # 🌱 NOUVEAU : Enregistrer les impacts EcoLogits
        llm_instance = test_chain.llm
        impacts = llm_instance.get_last_impacts()
        if impacts:
            tracker.record_impact(
                mode="test",
                username=username,
                energy=impacts["energy"]["value"],
                gwp=impacts["gwp"]["value"],
                model=LLM_MODEL,
                operation="grade"
            )

        try:
            save_result(username, question, answer, grading_json, expected_answer, key_points)
        except Exception as db_err:
            print("Database error:", db_err)
        
        print("Grading result:", grading_json)
        return grading_json

    except Exception as e:
        print("Error in grade_answer:", e)
        import traceback
        traceback.print_exc()
        return {
            "Section": "Error",
            "Question": question,
            "Answer": answer,
            "expected_answer": expected_answer,
            "key_points": key_points,
            "grade": 0,
            "scores": {
                "Key Points": 0,
                "Expected Match": 0,
                "Incorrect Facts": 0,
                "Structure": 0
            },
            "advice": "Internal error."
        }