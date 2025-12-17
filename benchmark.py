# Standalone Benchmark for Teach Mode
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import sys

from collections import defaultdict
from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader


# ============================================================================
# PATH SETUP - Auto-detect project root
# ============================================================================
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try to find project root (where faiss_index exists)
def find_project_root():
    """Search for project root by looking for faiss_index folder"""
    current = SCRIPT_DIR
    for _ in range(5):  # Search up to 5 levels up
        if os.path.exists(os.path.join(current, 'faiss_index')):
            return current
        parent = os.path.dirname(current)
        if parent == current:  # Reached filesystem root
            break
        current = parent
    return SCRIPT_DIR  # Fallback to script directory

PROJECT_ROOT = find_project_root()
print(f"Project root detected: {PROJECT_ROOT}")

# Change to project root
os.chdir(PROJECT_ROOT)

# Load environment variables
load_dotenv(override=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
PDF_PATH = "data/Atlas.pdf"
LLM_MODEL = "mistral-small-latest"
EMBED_MODEL = "mistral-embed"

if not os.path.exists(PDF_PATH):
    print(f"WARNING: PDF not found at {PDF_PATH}")
    print(f"PDF is only needed for Advanced RAG (BM25 setup)")
    sys.exit(1)

print(f"✓ PDF found at: {os.path.abspath(PDF_PATH)}")
print(f"✓ Configuration loaded\n")

# ============================================================================
# EMBEDDED FUNCTIONS FROM app/tools/rag.py
# ============================================================================
embeddings = MistralAIEmbeddings(model=EMBED_MODEL)


def build_vectorstore(documents, faiss_path, chunk_size=None, chunk_overlap=None):
    """
    Create a new FAISS vector database from documents.
    - If chunk_size/chunk_overlap provided, splits documents first
    - Converts each document chunk into a vector embedding using Mistral's embedding model
    - Stores these embeddings in a FAISS index for fast similarity search
    - Saves the index to disk so it can be reused without rebuilding
    - Returns the vectorstore object for immediate use
    
    Args:
        documents: List of documents (will be split if chunk_size provided)
        faiss_path: Path where the FAISS index should be saved
        chunk_size: Optional chunk size for splitting (uses documents as-is if None)
        chunk_overlap: Optional chunk overlap for splitting
    
    Returns:
        FAISS vectorstore object
    """
    # Split documents if chunk parameters provided
    if chunk_size is not None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap or 0
        )
        documents = splitter.split_documents(documents)
        print(f"Split into {len(documents)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    
    store = FAISS.from_documents(documents, embeddings)
    store.save_local(faiss_path)
    print(f"Vectorstore saved to {faiss_path}")
    return store

def format_docs(docs):
    """Format retrieved documents with page numbers"""
    return "\n\n".join(f"[Page {d.metadata.get('page','N/A')}] {d.page_content}" for d in docs)

# ============================================================================
# BENCHMARK QUESTIONS
# ============================================================================
benchmark_questions = {
    "points_chauds": [
        "Quels sont les principaux enjeux géopolitiques dans la région du Caucase ?",
        "Décrivez les tensions actuelles autour de Taïwan et leurs implications régionales.",
        "Quel est l'état actuel du conflit israélo-palestinien ?",
        "Comment la guerre en Ukraine affecte-t-elle l'équilibre géopolitique mondial ?",
    ],
    "ukraine": [
        "Quelles sont les forces et faiblesses militaires de la Russie dans le conflit ukrainien ?",
        "Comment l'Europe a-t-elle réagi face à l'invasion russe de l'Ukraine ?",
        "Quel est l'impact économique des sanctions contre la Russie ?",
        "La Russie peut-elle gagner la guerre en Ukraine ?",
    ],
    "indopacifique": [
        "Pourquoi l'Indopacifique est-il devenu un théâtre majeur des rivalités mondiales ?",
        "Quel est le rôle de la Chine dans la région indopacifique ?",
        "Comment les États-Unis répondent-ils aux ambitions chinoises en Asie ?",
        "Quels sont les enjeux maritimes en mer de Chine méridionale ?",
    ],
    "moyen_orient": [
        "Pourquoi la Syrie est-elle considérée comme une guerre inachevée ?",
        "Quelle est la situation politique actuelle en Iran ?",
        "Comment la Turquie sous Erdogan influence-t-elle la région ?",
        "Quels sont les défis géopolitiques au Moyen-Orient ?",
    ],
    "afrique": [
        "L'instabilité gagne-t-elle l'Afrique et pourquoi ?",
        "Qu'est-ce que l'arc de crise sahélien ?",
        "Quel est le rôle du groupe Wagner en Afrique ?",
        "Quels sont les principaux conflits actuels en Afrique ?",
    ],
    "grands_enjeux_2024": [
        "Quel est le rôle de l'OTAN 75 ans après sa création ?",
        "Comment la désinformation menace-t-elle les démocraties modernes ?",
        "Comment les Jeux olympiques peuvent-ils servir d'instrument de soft power ?",
        "Quelle est la situation politique actuelle en Amérique latine ?",
    ],
    "dissuasion_nucleaire": [
        "La dissuasion nucléaire a-t-elle encore un sens aujourd'hui ?",
        "Quels pays possèdent l'arme nucléaire ?",
        "Quels sont les risques liés à la prolifération nucléaire ?",
        "Comment fonctionne l'équilibre de la terreur nucléaire ?",
    ],
    "inegalites": [
        "Quels sont les principaux types d'inégalités dans le monde ?",
        "Comment résorber les inégalités économiques mondiales ?",
        "Quel est le lien entre inégalités et crises ?",
        "Le droit à l'avortement est-il menacé dans le monde ?",
    ],
    "ressources": [
        "Quelles sont les principales guerres pour les ressources ?",
        "Comment éradiquer la faim dans le monde ?",
        "Quels sont les enjeux liés à l'accès à l'eau ?",
        "Y a-t-il une compétition mondiale pour les ressources naturelles ?",
    ],
    "retour_histoire": [
        "Quelles sont les conséquences de l'annexion de la Crimée par la Russie en 2014 ?",
        "Expliquez l'importance de l'abolition de l'esclavage il y a 230 ans.",
        "Quels événements ont conduit à la libération de Paris il y a 80 ans ?",
        "Quelle était la signification du Congrès de Vienne il y a 210 ans ?",
    ],
    "empire_colonial": [
        "Comment s'est terminé l'Empire colonial espagnol il y a 200 ans ?",
        "Quelles furent les causes de l'indépendance des colonies espagnoles ?",
        "Quel a été l'impact des indépendances latino-américaines ?",
        "Qui étaient les principaux leaders des mouvements d'indépendance ?",
    ],
    "jeux_olympiques_histoire": [
        "Comment sont nés les Jeux olympiques modernes il y a 130 ans ?",
        "Qui a relancé les Jeux olympiques à l'époque moderne ?",
        "Quelle était la vision de Pierre de Coubertin ?",
        "Comment les JO ont-ils évolué depuis leur renaissance ?",
    ],
    "et_demain": [
        "Quels sont les défis liés à la transition énergétique pour atteindre les objectifs climatiques ?",
        "Comment la population mondiale devrait-elle évoluer d'ici 2050 ?",
        "Quels risques la géoingénierie présente-t-elle comme solution au réchauffement climatique ?",
        "Pourquoi la protection des océans est-elle cruciale pour l'avenir ?",
    ],
    "demographie": [
        "Combien d'humains y aura-t-il sur Terre en 2050 ?",
        "Où se concentrera la population mondiale dans le futur ?",
        "Quels sont les défis du vieillissement démographique ?",
        "Comment l'humanité sera-t-elle concentrée dans les pays du Sud ?",
    ],
    "transition_energetique": [
        "Pourquoi la transition énergétique est-elle urgente ?",
        "Quelles sont les principales sources d'énergie renouvelable ?",
        "Quels obstacles empêchent la transition énergétique ?",
        "Comment réduire notre dépendance aux énergies fossiles ?",
    ],
    "ville_futur": [
        "Comment devrons-nous vivre la ville différemment à l'avenir ?",
        "Un monde sans voitures est-il possible ?",
        "Quels sont les enjeux de l'urbanisation croissante ?",
        "Comment rendre les villes plus durables ?",
    ],
    "agriculture": [
        "Comment cultiver autrement pour nourrir 10 milliards d'humains ?",
        "Quels sont les avantages d'une agriculture plus bio ?",
        "Comment l'agriculture intensive affecte-t-elle l'environnement ?",
        "Quelles innovations agricoles pour demain ?",
    ],
    "environnement": [
        "Les forêts vont-elles disparaître ou migrer ?",
        "Comment protéger les mers et les océans ?",
        "Quels sont les impacts du réchauffement sur la biodiversité ?",
        "Pourquoi la forêt est-elle en crise ?",
    ],
    "crises_urgence": [
        "Quels types de catastrophes naturelles sont amplifiés par le réchauffement climatique ?",
        "Comment les inégalités influencent-elles l'exposition aux risques environnementaux ?",
        "Quels sont les risques épidémiques liés au réchauffement climatique ?",
        "Quelles politiques la France a-t-elle mises en place face aux risques naturels ?",
    ],
    "catastrophes_naturelles": [
        "Comment passe-t-on de l'aléa au risque pour les catastrophes naturelles ?",
        "Quels sont les types de catastrophes naturelles les plus fréquents ?",
        "Comment prévenir les catastrophes naturelles ?",
        "Quel est l'impact du réchauffement sur les catastrophes ?",
    ],
    "risques_eau": [
        "Quels sont les risques liés à l'eau dans le monde ?",
        "Y a-t-il une crise mondiale de l'eau ?",
        "Comment gérer les inondations et les sécheresses ?",
        "Quelles régions sont les plus vulnérables aux risques liés à l'eau ?",
    ],
}

all_questions = [q for topic_questions in benchmark_questions.values() for q in topic_questions]

# ============================================================================
# REFERENCE ANSWERS
# ============================================================================
reference_answers = {
    "Quels sont les principaux enjeux géopolitiques dans la région du Caucase ?":
        "Le Caucase est une zone charnière entre Europe, Russie, Moyen-Orient et Asie centrale, marquée par des rivalités d'influence (Russie, Turquie, Iran, Occident), des conflits autour des frontières et des minorités, ainsi que par des enjeux énergétiques et de corridors de transport.",
    "Décrivez les tensions actuelles autour de Taïwan et leurs implications régionales.":
        "Les tensions autour de Taïwan opposent la Chine, qui revendique l'île, aux États-Unis et à leurs partenaires qui soutiennent de facto son autonomie, ce qui alimente une course aux armements en Asie de l'Est et renforce les alliances régionales de sécurité.",
    "Quel est l'état actuel du conflit israélo-palestinien ?":
        "Le conflit israélo-palestinien reste caractérisé par l'absence de solution politique négociée, la poursuite de la colonisation, des cycles récurrents de violences, et une crise humanitaire grave dans les territoires palestiniens.",
    "Comment la guerre en Ukraine affecte-t-elle l'équilibre géopolitique mondial ?":
        "La guerre en Ukraine a renforcé la cohésion des alliés occidentaux, accentué la confrontation avec la Russie, accéléré le réalignement énergétique européen et rapproché davantage Moscou de puissances non occidentales comme la Chine, ce qui recompose l'équilibre mondial.",
    "Quelles sont les forces et faiblesses militaires de la Russie dans le conflit ukrainien ?":
        "La Russie dispose d'un important arsenal, d'une base industrielle de défense et de la supériorité en artillerie, mais elle souffre de contraintes logistiques, de pertes humaines et matérielles élevées, de problèmes de commandement et de moral ainsi que d'une difficulté à mener des opérations combinées efficaces.",
    "Comment l'Europe a-t-elle réagi face à l'invasion russe de l'Ukraine ?":
        "L'Europe a répondu par des sanctions économiques massives contre la Russie, une aide militaire et financière significative à l'Ukraine, le renforcement de l'OTAN et une réorientation rapide de sa politique énergétique pour réduire sa dépendance aux hydrocarbures russes.",
    "Quel est l'impact économique des sanctions contre la Russie ?":
        "Les sanctions ont restreint l'accès de la Russie aux marchés, aux technologies et aux financements occidentaux, provoqué une réorientation forcée de ses exportations vers d'autres partenaires et entraîné des effets de contournement, tandis qu'elles ont aussi généré des coûts pour certaines économies européennes.",
    "La Russie peut-elle gagner la guerre en Ukraine ?":
        "L'issue du conflit reste incertaine et dépend de nombreux facteurs militaires, politiques et économiques, mais la Russie fait face à des contraintes structurelles et à une forte résistance ukrainienne soutenue par l'aide occidentale, ce qui rend improbable une victoire rapide et totale.",
    "Pourquoi l'Indopacifique est-il devenu un théâtre majeur des rivalités mondiales ?":
        "L'Indopacifique concentre une grande part de la population mondiale, du commerce maritime et des chaînes de valeur, et voit la montée en puissance de la Chine face aux États-Unis, ce qui en fait un espace clé de compétition stratégique, technologique et navale.",
    "Quel est le rôle de la Chine dans la région indopacifique ?":
        "La Chine cherche à y affirmer son statut de grande puissance, à sécuriser ses routes maritimes et ses approvisionnements, à projeter sa puissance militaire et à étendre son influence économique et diplomatique à travers des investissements et des partenariats.",
    "Comment les États-Unis répondent-ils aux ambitions chinoises en Asie ?":
        "Les États-Unis renforcent leurs alliances et partenariats (Japon, Corée du Sud, Australie, Inde, ASEAN), augmentent leur présence militaire, développent la coopération technologique et sécuritaire, et promeuvent des initiatives économiques alternatives pour contenir l'influence chinoise.",
    "Quels sont les enjeux maritimes en mer de Chine méridionale ?":
        "La mer de Chine méridionale concentre des voies maritimes essentielles, des ressources halieutiques et énergétiques, et fait l'objet de revendications territoriales concurrentes, notamment de la Chine, ce qui crée des tensions autour de la liberté de navigation et du droit international de la mer.",
    "Pourquoi la Syrie est-elle considérée comme une guerre inachevée ?":
        "Le régime a repris le contrôle d'une grande partie du territoire mais le pays reste fragmenté, avec des zones tenues par d'autres acteurs, une présence de forces étrangères, une crise humanitaire profonde et une absence de solution politique globale.",
    "Quelle est la situation politique actuelle en Iran ?":
        "L'Iran est dirigé par un régime théocratique qui fait face à des tensions internes récurrentes, à des sanctions internationales, à des difficultés économiques et à des rivalités régionales, tout en poursuivant une politique d'influence au Moyen-Orient.",
    "Comment la Turquie sous Erdogan influence-t-elle la région ?":
        "La Turquie mène une politique étrangère ambitieuse, combinant interventions militaires, soutien à certains groupes politiques, diplomatie énergétique et rôle de médiateur, ce qui lui permet de peser en Syrie, dans le Caucase, en Méditerranée orientale et au-delà.",
    "Quels sont les défis géopolitiques au Moyen-Orient ?":
        "La région est marquée par des rivalités entre puissances régionales, des conflits armés, la question palestinienne, les enjeux énergétiques, la fragmentation étatique et les tensions confessionnelles, le tout sur fond de transitions politiques inachevées.",
    "L'instabilité gagne-t-elle l'Afrique et pourquoi ?":
        "De nombreux pays africains connaissent des coups d'État, des insurrections armées et des tensions politiques, alimentés par la pauvreté, la faiblesse des institutions, la compétition pour les ressources et les ingérences extérieures.",
    "Qu'est-ce que l'arc de crise sahélien ?":
        "L'arc de crise sahélien désigne la bande allant du Sahel à la Corne de l'Afrique, marquée par l'insécurité, les groupes armés, le terrorisme, les trafics, la fragilité des États et les effets du changement climatique sur les sociétés.",
    "Quel est le rôle du groupe Wagner en Afrique ?":
        "Le groupe Wagner, ou ses structures successeures, agit comme instrument d'influence d'intérêts russes dans plusieurs pays africains en fournissant des services de sécurité, en soutenant certains régimes et en obtenant des contreparties minières ou politiques.",
    "Quels sont les principaux conflits actuels en Afrique ?":
        "On trouve des conflits armés ou insurrections notables au Sahel, en Afrique centrale, dans la Corne de l'Afrique et dans certaines régions côtières, souvent mêlant enjeux politiques, identitaires, économiques et environnementaux.",
}

# ============================================================================
# Simple RAG, with different values of CHUNK_SIZE and CHUNK_OVERLAP
# ============================================================================

loader = PyMuPDFLoader(PDF_PATH)
docs = loader.load()

# Configuration 1: Small chunks
CHUNK_SIZE_1 = 250
CHUNK_OVERLAP_1 = 25
FAISS_PATH_1 = "faiss_index_chunk250"

if os.path.exists(FAISS_PATH_1):
    print(f"Loading existing vectorstore from {FAISS_PATH_1}")
    vectorstore_1 = FAISS.load_local(FAISS_PATH_1, embeddings, allow_dangerous_deserialization=True)
else:
    print(f"Building new vectorstore at {FAISS_PATH_1}")
    vectorstore_1 = build_vectorstore(docs, FAISS_PATH_1, chunk_size=CHUNK_SIZE_1, chunk_overlap=CHUNK_OVERLAP_1)

retriever_1 = vectorstore_1.as_retriever(search_type='similarity', search_kwargs={'k': 7})

# Configuration 2: Medium chunks
CHUNK_SIZE_2 = 500
CHUNK_OVERLAP_2 = 50
FAISS_PATH_2 = "faiss_index_chunk500"

if os.path.exists(FAISS_PATH_2):
    print(f"Loading existing vectorstore from {FAISS_PATH_2}")
    vectorstore_2 = FAISS.load_local(FAISS_PATH_2, embeddings, allow_dangerous_deserialization=True)
else:
    print(f"Building new vectorstore at {FAISS_PATH_2}")
    vectorstore_2 = build_vectorstore(docs, FAISS_PATH_2, chunk_size=CHUNK_SIZE_2, chunk_overlap=CHUNK_OVERLAP_2)

retriever_2 = vectorstore_2.as_retriever(search_type='similarity', search_kwargs={'k': 7})

# Configuration 3: Large chunks
CHUNK_SIZE_3 = 1000
CHUNK_OVERLAP_3 = 100
FAISS_PATH_3 = "faiss_index_chunk1000"

if os.path.exists(FAISS_PATH_3):
    print(f"Loading existing vectorstore from {FAISS_PATH_3}")
    vectorstore_3 = FAISS.load_local(FAISS_PATH_3, embeddings, allow_dangerous_deserialization=True)
else:
    print(f"Building new vectorstore at {FAISS_PATH_3}")
    vectorstore_3 = build_vectorstore(docs, FAISS_PATH_3, chunk_size=CHUNK_SIZE_3, chunk_overlap=CHUNK_OVERLAP_3)

retriever_3 = vectorstore_3.as_retriever(search_type='similarity', search_kwargs={'k': 7})

# LLM
llm = ChatMistralAI(model=LLM_MODEL, temperature=0)

# Persona prompt from project (modified to cite page numbers)
persona_prompt = ChatPromptTemplate.from_messages([
    ("system",
            """Tu es un professeur d'histoire-géographie expérimenté (20 ans d'enseignement).
    Tu aides un élève en difficulté en expliquant clairement, sans jamais inventer d'informations.

    PRIORITÉS (dans cet ordre) :
    1. Exactitude : ne répondre qu'avec les informations présentes dans le contexte fourni.
    2. Rigueur : si le contexte ne contient pas la réponse, dis-le explicitement.
    3. Pertinence : si la question est hors programme ou sans lien avec le chapitre, indique-le clairement.
    4. Style : réponses courtes, claires, structurées en paragraphes avec sauts de ligne si nécessaire.
    5. Citations : TOUJOURS citer la page source sous forme [Page X] après chaque fait mentionné.

    RÈGLES :
    - N'utilise comme source que : (a) le contexte, (b) l'historique de conversation, uniquement pour le fil logique, jamais comme source factuelle.
    - Ne mentionne jamais l'existence du contexte, de règles ou de contraintes.
    - Pour l'élève, le contexte correspond simplement à son manuel "Le Grand Atlas".
    - Si une information n'apparaît nulle part dans le contexte, invite l'élève à se référer à son professeur.
    - Si l'élève fait une erreur factuelle, corrige-le avec bienveillance.
    - Ton ton est encourageant mais professionnel : pas d'humour, pas de familiarité.
    - Explique de manière fluide et pédagogique, en évitant les phrases trop longues.
    - IMPORTANT : Cite systématiquement la page après chaque information factuelle en utilisant le format [Page X].
      Le contexte fourni contient déjà les numéros de page sous forme [Page X] au début de chaque extrait.

    Chapitre du cours : {chapter_context}

    Historique de conversation dans ce chapitre :
    {history}


    """),
    ("human",
     "Question: {question}\n\nContexte:\n{context}")
])

def create_rag_chain(retriever):
    """Factory function to create a RAG chain with a specific retriever."""
    return (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough(),
            "chapter_context": lambda _: "Chapitre général",
            "history": lambda _: "Aucune conversation précédente"
        } 
        | persona_prompt
        | llm
        | StrOutputParser()
    )

# Create three chains
rag_chain_1 = create_rag_chain(retriever_1)
rag_chain_2 = create_rag_chain(retriever_2)
rag_chain_3 = create_rag_chain(retriever_3)

def ask_basic_rag_1(question: str) -> str:
    """Ask using RAG with chunk size 250."""
    return rag_chain_1.invoke(question)

def ask_basic_rag_2(question: str) -> str:
    """Ask using RAG with chunk size 500."""
    return rag_chain_2.invoke(question)

def ask_basic_rag_3(question: str) -> str:
    """Ask using RAG with chunk size 1000."""
    return rag_chain_3.invoke(question)



# ============================================================================
# ADVANCED RAG WITH ENSEMBLE RETRIEVER - FACTORY APPROACH
# ============================================================================



def _doc_key(doc: Document) -> str:
    return doc.page_content + "||" + str(sorted(doc.metadata.items()))

def weighted_rrf(doc_lists: List[List[Document]], weights: List[float], top_k: int = 4, c: int = 60):
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

def create_ensemble_retriever(chunk_size, chunk_overlap, dense_vectorstore):
    """Factory function to create an ensemble retriever with BM25 + dense retrieval."""
    # Load and split documents for BM25
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)
    
    # Prepare BM25 retriever
    texts_for_bm25 = [d.page_content for d in split_docs]
    bm25 = BM25Retriever.from_texts(texts_for_bm25, metadatas=[d.metadata for d in split_docs])
    bm25.k = 7
    
    # Dense retriever
    dense_retriever = dense_vectorstore.as_retriever(search_kwargs={'k': 7})
    
    # Combine with ensemble
    ensemble_retriever = SimpleEnsembleRetriever(
        retrievers=[bm25, dense_retriever],
        weights=[0.1, 0.9],
        k=7
    )
    
    return ensemble_retriever

# Create three ensemble retrievers
ensemble_retriever_1 = create_ensemble_retriever(250, 25, vectorstore_1)
ensemble_retriever_2 = create_ensemble_retriever(500, 50, vectorstore_2)
ensemble_retriever_3 = create_ensemble_retriever(1000, 100, vectorstore_3)

advanced_rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
            """Tu es un professeur d'histoire-géographie expérimenté (20 ans d'enseignement).
    Tu aides un élève en difficulté en expliquant clairement, sans jamais inventer d'informations.

    PRIORITÉS (dans cet ordre) :
    1. Exactitude : ne répondre qu'avec les informations présentes dans le contexte fourni.
    2. Rigueur : si le contexte ne contient pas la réponse, dis-le explicitement.
    3. Pertinence : si la question est hors programme ou sans lien avec le chapitre, indique-le clairement.
    4. Style : réponses courtes, claires, structurées en paragraphes avec sauts de ligne si nécessaire.
    5. Citations : TOUJOURS citer la page source sous forme [Page X] après chaque fait mentionné.

    RÈGLES :
    - N'utilise comme source que : (a) le contexte, (b) l'historique de conversation, uniquement pour le fil logique, jamais comme source factuelle.
    - Ne mentionne jamais l'existence du contexte, de règles ou de contraintes.
    - Pour l'élève, le contexte correspond simplement à son manuel "Le Grand Atlas".
    - Si une information n'apparaît nulle part dans le contexte, invite l'élève à se référer à son professeur.
    - Si l'élève fait une erreur factuelle, corrige-le avec bienveillance.
    - Ton ton est encourageant mais professionnel : pas d'humour, pas de familiarité.
    - Explique de manière fluide et pédagogique, en évitant les phrases trop longues.
    - IMPORTANT : Cite systématiquement la page après chaque information factuelle en utilisant le format [Page X].
      Le contexte fourni contient déjà les numéros de page sous forme [Page X] au début de chaque extrait.

    Chapitre du cours : {chapter_context}

    Historique de conversation dans ce chapitre :
    {history}


    """),
    ("human",
     "Question: {question}\n\nContexte:\n{context}")
])

def create_advanced_rag_chain(ensemble_retriever):
    """Factory function to create an advanced RAG chain with ensemble retriever."""
    return (
        {
            "context": ensemble_retriever | format_docs, 
            "question": RunnablePassthrough(),
            "chapter_context": lambda _: "Chapitre général",
            "history": lambda _: "Aucune conversation précédente"
        } 
        | advanced_rag_prompt
        | llm
        | StrOutputParser()
    )

# Create three advanced RAG chains
advanced_rag_chain_1 = create_advanced_rag_chain(ensemble_retriever_1)
advanced_rag_chain_2 = create_advanced_rag_chain(ensemble_retriever_2)
advanced_rag_chain_3 = create_advanced_rag_chain(ensemble_retriever_3)

def ask_advanced_rag_1(question: str) -> str:
    """Ask using advanced RAG with chunk size 250."""
    return advanced_rag_chain_1.invoke(question)

def ask_advanced_rag_2(question: str) -> str:
    """Ask using advanced RAG with chunk size 500."""
    return advanced_rag_chain_2.invoke(question)

def ask_advanced_rag_3(question: str) -> str:
    """Ask using advanced RAG with chunk size 1000."""
    return advanced_rag_chain_3.invoke(question)




# ============================================================================
# JUDGE EVALUATION SYSTEM
# ============================================================================
import json
import re

judge_prompt = PromptTemplate.from_template(
    """Tu es un évaluateur expert de systèmes de questions-réponses.
Évalue la RÉPONSE CANDIDATE en la comparant à la RÉPONSE DE RÉFÉRENCE.

QUESTION: {question}

RÉPONSE DE RÉFÉRENCE (ground truth):
{reference_answer}

RÉPONSE CANDIDATE (à évaluer):
{candidate_answer}

CRITÈRES D'ÉVALUATION (note de 1 à 5):
1. ACCURACY (exactitude): Les faits sont-ils corrects?
2. COMPLETENESS (complétude): Tous les points importants sont-ils couverts?
3. CITATION_QUALITY (qualité des citations): Les sources sont-elles bien citées?
4. RELEVANCE (pertinence): La réponse est-elle pertinente pour la question?

INSTRUCTIONS:
- Note chaque critère de 1 (très mauvais) à 5 (excellent)
- Donne une note globale (OVERALL) de 1 à 5
- Fournis une brève justification

FORMAT DE RÉPONSE: Réponds UNIQUEMENT avec un objet JSON valide (sans markdown):
{{
    "accuracy": <score>,
    "completeness": <score>,
    "citation_quality": <score>,
    "relevance": <score>,
    "overall": <score>,
    "justification": "<ton explication en 2-3 phrases>"
}}
"""
)

judge_llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
judge_chain = judge_prompt | judge_llm | StrOutputParser()

def parse_judge_scores(judge_output: str) -> dict:
    """Parse judge output to extract scores from JSON."""
    try:
        # Try to parse as JSON directly
        scores = json.loads(judge_output)
        # Ensure all required keys exist with default values
        default_scores = {
            'accuracy': 0,
            'completeness': 0,
            'citation_quality': 0,
            'relevance': 0,
            'overall': 0,
            'justification': ''
        }
        default_scores.update(scores)
        return default_scores
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', judge_output, re.DOTALL)
        if json_match:
            try:
                scores = json.loads(json_match.group(1))
                default_scores = {
                    'accuracy': 0,
                    'completeness': 0,
                    'citation_quality': 0,
                    'relevance': 0,
                    'overall': 0,
                    'justification': ''
                }
                default_scores.update(scores)
                return default_scores
            except json.JSONDecodeError:
                pass
        
        # Return default scores if parsing fails
        return {
            'accuracy': 0,
            'completeness': 0,
            'citation_quality': 0,
            'relevance': 0,
            'overall': 0,
            'justification': 'Error parsing judge output'
        }

def evaluate_answer(question: str, reference_answer: str, candidate_answer: str) -> dict:
    """Evaluate a candidate answer against a reference answer."""
    judge_output = judge_chain.invoke({
        "question": question,
        "reference_answer": reference_answer,
        "candidate_answer": candidate_answer
    })
    return parse_judge_scores(judge_output)


# ============================================================================
# COMPLETE EVALUATION RUNNER
# ============================================================================
import pandas as pd
import time

def run_complete_evaluation():
    """Run complete evaluation of all configurations on all questions."""
    print("Running complete evaluation...")
    
    # Filter to only questions with reference answers
    questions_to_eval = [q for q in all_questions if q in reference_answers]
    
    print(f"Total questions: {len(all_questions)}")
    print(f"Questions with reference answers: {len(questions_to_eval)}")
    print(f"Will evaluate: {len(questions_to_eval)} questions\n")

    results = []

    configurations = {
        'Basic RAG (chunk 250)': ask_basic_rag_1,
        'Basic RAG (chunk 500)': ask_basic_rag_2,
        'Basic RAG (chunk 1000)': ask_basic_rag_3,
        'Advanced RAG (chunk 250)': ask_advanced_rag_1,
        'Advanced RAG (chunk 500)': ask_advanced_rag_2,
        'Advanced RAG (chunk 1000)': ask_advanced_rag_3
    }

    total_evals = len(questions_to_eval) * len(configurations)
    current_eval = 0

    for question in questions_to_eval:
        reference = reference_answers[question]
        
        # Find topic
        topic = 'other'
        for t, qs in benchmark_questions.items():
            if question in qs:
                topic = t
                break
        
        for config_name, ask_func in configurations.items():
            current_eval += 1
            print(f"[{current_eval}/{total_evals}] {config_name}: {question[:50]}...")
            
            try:
                # Generate answer
                candidate = ask_func(question)
                
                # Evaluate
                scores = evaluate_answer(question, reference, candidate)
                
                # Store only scores, topic, and configuration
                results.append({
                    'topic': topic,
                    'configuration': config_name,
                    'accuracy': scores['accuracy'],
                    'completeness': scores['completeness'],
                    'citation_quality': scores['citation_quality'],
                    'relevance': scores['relevance'],
                    'overall': scores['overall']
                })
                
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    'topic': topic,
                    'configuration': config_name,
                    'accuracy': 0,
                    'completeness': 0,
                    'citation_quality': 0,
                    'relevance': 0,
                    'overall': 0
                })

    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Calculate averages by configuration and topic
    df_summary = df_results.groupby(['configuration', 'topic'])[['accuracy', 'completeness', 
                                                                   'citation_quality', 'relevance', 
                                                                   'overall']].mean().reset_index()
    
    print("\n✓ Evaluation complete!")
    print(f"Total evaluations: {len(df_results)}")
    
    return df_summary

# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # Check if user wants to run full evaluation
    run_full_eval = '--full' in sys.argv
    
    if run_full_eval:
        # Run complete evaluation on all questions
        df_summary = run_complete_evaluation()
        
        # Save results
        df_summary.to_csv('benchmark_results_summary.csv', index=False)
        print("\n✓ Results saved to 'benchmark_results_summary.csv'")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("OVERALL PERFORMANCE BY CONFIGURATION")
        print("="*80)
        
        overall_summary = df_summary.groupby('configuration')[['accuracy', 'completeness', 
                                                                'citation_quality', 'relevance', 
                                                                'overall']].mean()
        print(overall_summary.round(2))
        print("\n")
        
        # Best configuration overall
        best_config = overall_summary['overall'].idxmax()
        best_score = overall_summary['overall'].max()
        print(f"Best Configuration: {best_config} (Overall Score: {best_score:.2f})")
        
        print("\n" + "="*80)
        print("PERFORMANCE BY TOPIC")
        print("="*80)
        topic_summary = df_summary.pivot_table(values='overall', index='topic', columns='configuration')
        print(topic_summary.round(2))
    else:
        # Run quick test on single question
        test_q = "Comment la guerre en Ukraine affecte-t-elle l'équilibre géopolitique mondial ?"
        
        print("="*80)
        print("QUICK TEST MODE")
        print("(Use --full flag to run complete evaluation)")
        print("="*80)
        
        print("\n" + "="*80)
        print("TEST: Basic RAG (chunk 250)")
        print("="*80)
        print(f"Q: {test_q}")
        answer_1 = ask_basic_rag_1(test_q)
        print(f"A: {answer_1}")
        
        print("\n" + "="*80)
        print("TEST: Basic RAG (chunk 500)")
        print("="*80)
        print(f"Q: {test_q}")
        answer_2 = ask_basic_rag_2(test_q)
        print(f"A: {answer_2}")
        
        print("\n" + "="*80)
        print("TEST: Advanced RAG (chunk 500)")
        print("="*80)
        print(f"Q: {test_q}")
        advanced_answer = ask_advanced_rag_2(test_q)
        print(f"A: {advanced_answer}")
        
        # Test the judge evaluation
        if test_q in reference_answers:
            print("\n" + "="*80)
            print("TEST: Judge Evaluation (Basic RAG chunk 500)")
            print("="*80)
            test_evaluation = evaluate_answer(
                test_q,
                reference_answers[test_q],
                answer_2
            )
            print("Evaluation scores:")
            for key, value in test_evaluation.items():
                print(f"  {key}: {value}")

