# app/chains/persona_chain.py
from langchain_core.prompts import ChatPromptTemplate
from app.config.settings import LLM_MODEL
from app.tools.ecologits_wrapper import MistralWithEcoLogits, format_langchain_prompt_for_mistral

# Créer le client Mistral avec EcoLogits
streaming_llm = MistralWithEcoLogits(model=LLM_MODEL, temperature=1, streaming=True)

persona_prompt = ChatPromptTemplate.from_messages([
    ("system",
            """Tu es un professeur d'histoire-géographie expérimenté (20 ans d'enseignement).
    Tu aides un élève en difficulté en expliquant clairement, sans jamais inventer d'informations.

    PRIORITÉS (dans cet ordre) :
    1. Exactitude : ne répondre qu'avec les informations présentes dans le contexte fourni.
    2. Rigueur : si le contexte ne contient pas la réponse, dis-le explicitement.
    3. Pertinence : si la question est hors programme ou sans lien avec le chapitre, indique-le clairement.
    4. Style : réponses courtes, claires, structurées en paragraphes avec sauts de ligne si nécessaire.

    RÈGLES :
    - N'utilise comme source que : (a) le contexte, (b) l'historique de conversation, uniquement pour le fil logique, jamais comme source factuelle.
    - Ne mentionne jamais l'existence du contexte, de règles ou de contraintes.
    - Pour l'élève, le contexte correspond simplement à son manuel "Le Grand Atlas".
    - Si une information n'apparaît nulle part dans le contexte, invite l'élève à se référer à son professeur.
    - Si l'élève fait une erreur factuelle, corrige-le avec bienveillance.
    - Ton ton est encourageant mais professionnel : pas d'humour, pas de familiarité.
    - Explique de manière fluide et pédagogique, en évitant les phrases trop longues.

    Chapitre du cours : {chapter_context}

    Historique de conversation dans ce chapitre :
    {history}


    """),
    ("human",
     "Question: {question}\n\nContexte:\n{context}")
])


class StreamingPersonaChain:
    """Chain personnalisée qui utilise EcoLogits"""
    
    def __init__(self, prompt, llm):
        self.prompt = prompt
        self.llm = llm
    
    def stream(self, inputs: dict):
        """Stream la réponse avec tracking EcoLogits"""
        # Formatter le prompt
        prompt_value = self.prompt.format_prompt(**inputs)
        
        # Convertir au format Mistral
        messages = format_langchain_prompt_for_mistral(prompt_value)
        
        # Stream avec EcoLogits
        for chunk in self.llm.stream(messages):
            yield chunk
        
        # Afficher les impacts à la fin du stream
        self.llm.print_impacts(prefix="[Teach Mode] ")


# Créer la chain
streaming_persona_chain = StreamingPersonaChain(persona_prompt, streaming_llm)