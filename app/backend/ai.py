# backend/ai.py
import json
from dotenv import load_dotenv

from app.tools.database import save_result
load_dotenv()

from app.tools.rag import get_retriever, load_vectorstore
from app.tools.loaders import load_pdf
from app.tools.loaders import caption_images
from app.tools.rag import split_docs
from app.chains.persona_chain import streaming_persona_chain, persona_prompt
from app.chains.test_chain import generate_question_chain, test_chain
from app.config.settings import FAISS_PATH, PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from app.chains.theme_chain import theme_llm
from app.tools.database import save_teach_interaction
from langchain_core.prompts import ChatPromptTemplate
from app.tools.database import get_student_chapter_interactions
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensemble retriever components
from collections import defaultdict
from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def format_docs(docs):
    """Format retrieved documents with page numbers"""
    return "\n\n".join(f"[Page {d.metadata.get('page','N/A')}] {d.page_content}" for d in docs)


# ============================================================================
# ENSEMBLE RETRIEVER IMPLEMENTATION
# ============================================================================

def _doc_key(doc: Document) -> str:
    return doc.page_content + "||" + str(sorted(doc.metadata.items()))

def weighted_rrf(doc_lists: List[List[Document]], weights: List[float], top_k: int = 7, c: int = 60):
    scores = defaultdict(float)
    doc_map = {}
    
    for i, docs in enumerate(doc_lists):
        w = weights[i] if i < len(weights) else 1.0
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            scores[key] += w / (c + rank)
            if key not in doc_map:
                doc_map[key] = doc
    
    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys[:top_k]]

class SimpleEnsembleRetriever(BaseRetriever, BaseModel):
    retrievers: List[BaseRetriever]
    weights: List[float]
    k: int = 7
    c: int = 60
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        lists = [r.invoke(query) for r in self.retrievers]
        return weighted_rrf(lists, self.weights, top_k=self.k, c=self.c)
    
    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        lists = []
        for r in self.retrievers:
            try:
                docs = await r.ainvoke(query)
            except Exception:
                docs = r.invoke(query)
            lists.append(docs)
        return weighted_rrf(lists, self.weights, top_k=self.k, c=self.c)


# ============================================================================
# INITIALIZE RETRIEVERS
# ============================================================================

print("Initializing AI backend…")

docs = load_pdf()
image_docs = caption_images()
chunks = split_docs(docs)
all_docs = chunks

# Build/load dense vectorstore
_ = get_retriever(all_docs, FAISS_PATH, search_k=7)

# Create BM25 retriever
loader = PyMuPDFLoader(PDF_PATH)
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
split_docs_for_bm25 = splitter.split_documents(raw_docs)

texts_for_bm25 = [d.page_content for d in split_docs_for_bm25]
bm25_retriever = BM25Retriever.from_texts(texts_for_bm25, metadatas=[d.metadata for d in split_docs_for_bm25])
bm25_retriever.k = 7

# Create dense retriever from vectorstore
dense_vectorstore = load_vectorstore(FAISS_PATH)
dense_retriever = dense_vectorstore.as_retriever(search_kwargs={'k': 7})

# Create ensemble retriever (BM25 + Dense)
retriever = SimpleEnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.50, 0.50],
    k=7
)

print("✓ Ensemble retriever initialized (BM25 + Dense retrieval)")


# ============================================================================
# REST OF THE CODE REMAINS UNCHANGED
# ============================================================================

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
            previous_chapter = recent_history[0][1]
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
        
        if "OUI" in is_followup.upper() and previous_chapter != "Aucun":
            detected_chapter = previous_chapter
            reason = f"Suite logique détectée → même chapitre: {previous_chapter}"
        
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
    """Stream answer from the RAG/chat system token by token."""
    from app.tools.database import get_student_chapter_interactions
    
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

        # Retrieve context using ensemble retriever
        docs = retriever.invoke(question)
        context = format_docs(docs)
        
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
        
        token_count_input = 0
        token_count_output = 0
        
        for chunk in streaming_persona_chain.stream(stream_inputs):
            if hasattr(chunk, 'content'):
                content = chunk.content
                if content:
                    full_answer += content
                    yield content
            elif isinstance(chunk, str):
                if chunk:
                    full_answer += chunk
                    yield chunk
            
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata is not None:
                token_count_input = chunk.usage_metadata.get('input_tokens', 0)
                token_count_output = chunk.usage_metadata.get('output_tokens', 0)
        
        print(f"\n📊 Token usage - Input: {token_count_input}, Output: {token_count_output}")
        
        if full_answer:
            save_teach_interaction(username, chapter_context, question, full_answer, token_count_input, token_count_output)
                    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Error in ai_answer_stream: {e}")
        import traceback
        traceback.print_exc()
        yield error_msg
        
        if chapter_context or question:
            save_teach_interaction(username, chapter_context or "Error", question, error_msg, 0, 0)


def generate_test_question(criteria):
    """Generate a test question using document context (RAG) and student criteria."""
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
    """Grade a student's answer based on objective criteria."""
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

        try:
            save_result(username, question, answer, grading_json)
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