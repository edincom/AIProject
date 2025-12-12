    followup_prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un classifieur binaire. Ton rôle est de déterminer si la question actuelle est une suite logique de la conversation précédente.

        RÈGLE DE DÉCISION :
        - Réponds "OUI" uniquement si la question dépend explicitement du contenu précédent pour être comprise.
        - Réponds "NON" si la question est compréhensible seule, introduit un nouveau sujet, ou n’a pas de lien clair avec la conversation.
        - En cas de doute, réponds "NON".

        INDICATEURS DE SUIVI (OUI) :
        - Pronoms ou références sans antécédent dans la question (ex: il, elle, ça, cela, celui-ci, celui-là).
        - Questions incomplètes ou elliptiques ("Et lui ?", "Et après ?", "Pourquoi ça ?").
        - Demandes de clarification sur la réponse précédente.
        - Questions qui prolongent explicitement le même sujet.
        - Références implicites à un élément mentionné seulement dans la conversation précédente.

        INDICATEURS DE NON SUIVI (NON) :
        - Nouvelle thématique non liée au contenu précédent.
        - Question totalement autonome sémantiquement.
        - Reformulation générale sans lien précis avec la conversation.
        - Changement brutal de sujet.
        - Utilisation d'informations absentes de la conversation.

        CONTRAINTE DE SORTIE :
        Réponds STRICTEMENT par : OUI ou NON
        Sans ponctuation, sans justification, sans texte additionnel.

        ---
         
        Voici quelques exemùples sur lesquelles t'appuyer : 
        1. Suite logique par pronom référentiel :
        Conversation :
        A : Napoléon a été exilé à Sainte-Hélène.
        B : Et après, qu’est-ce qu’il a fait là-bas ?
        → OUI

        2. Poursuite implicite du même sujet :
        Conversation :
        A : La Première Guerre mondiale commence en 1914.
        B : Et ça dure combien de temps ?
        → OUI

        3. Demande de clarification :
        Conversation :
        A : La Renaissance commence en Italie.
        B : Pourquoi là-bas en premier ?
        → OUI

        4. Question elliptique / fragmentaire :
        Conversation :
        A : Le suffrage universel masculin apparaît en 1848.
        B : Et pour les femmes ?
        → OUI

        5. Référence à une structure mentionnée précédemment (liste, phases, etc.) :
        Conversation :
        A : La Révolution française comporte trois phases principales.
        B : Peux-tu m’expliquer la deuxième ?
        → OUI

        6. Confirmation / reformulation :
        Conversation :
        A : La guerre froide oppose principalement les États-Unis et l’URSS.
        B : Donc c’est une guerre indirecte, c’est ça ?
        → OUI

        7. Prolongement du sujet :
        Conversation :
        A : L’Union européenne compte 27 membres.
        B : Et ils ont tous les mêmes droits ?
        → OUI

        8. Suivi sur une partie implicite :
        Conversation :
        A : L’esclavage est aboli en 1848 en France.
        B : Et dans les colonies ?
        → OUI

        9. Clarification sur un terme imprécis :
        Conversation :
        A : La crise de 1929 entraîne une énorme hausse du chômage.
        B : Quand tu dis “énorme”, tu veux dire combien ?
        → OUI

        10. Suite logique par demande de cause :
        Conversation :
        A : La Chine devient une puissance majeure au XXᵉ siècle.
        B : À cause de quoi exactement ?
        → OUI



        Conversation récente :
        {conversation}

        Question actuelle : {question}

        Est-ce une suite logique ?""")
    ])




def is_followup_question(question, recent_history):
    """Determine if a question is a follow-up to recent conversation"""
    if not recent_history or len(recent_history) == 0:
        return False, None
    print("hello")
    from langchain_core.prompts import ChatPromptTemplate
    
    # Build recent conversation context
    conversation = "\n".join([
        f"Q: {h[2]}\nR: {h[3][:200]}..." for h in recent_history[:2]  # Last 2 interactions, truncated
    ])
    
    followup_prompt = ChatPromptTemplate.from_messages([
        ("system", """
    Tu es un classifieur binaire. Détermine si la question actuelle est une suite logique
    de la conversation précédente (dépend du contexte pour être comprise) ou une
    nouvelle question indépendante.

    RÈGLES DE DÉCISION :
    - OUI si la question NE PEUT PAS être correctement comprise sans référents,
    éléments ou informations présents dans la conversation précédente.
    - NON si la question est autonome, introduit un nouveau sujet, ou n'a pas besoin
    du contexte précédent.
    - En cas de doute → NON.

    INDICATEURS COURANTS (exemples explicites) :
    - Pronoms sans antécédent : il, elle, ils, elles, ça, cela, celui-ci, celui-là, son, sa, lui.
    - Questions elliptiques ou fragments : "Et après ?", "Et lui ?", "Et pour les femmes ?".
    - Demandes de clarification ciblées sur la réponse précédente : "Que veux-tu dire par X ?".
    - Référence chiffrée à une structure mentionnée avant : "la deuxième", "la troisième phase".
    - Référence à une entité, date ou événement présent uniquement dans l'historique.

    SORTIE ATTENDUE (strict) :
    - Deux lignes exactement.
    - Ligne 1 : OUI ou NON (MAJUSCULE).
    - Ligne 2 : RAISON: <phrase brève, 10–20 mots max> — indique l'indicateur précis qui motive la décision.
    - Aucune autre ligne, ponctuation, explication supplémentaire ou variation autorisée.

    CONTRAINTES :
    - Ne pas inventer d'informations.
    - Ne pas utiliser l'historique comme source factuelle si le contexte fourni est absent.
    """),
        ("human", """Conversation récente :
    {conversation}

    Question actuelle :
    {question}

    Décide : la question est-elle une suite logique ? Respecte strictement le format demandé.""")
    ])

    
    chain = followup_prompt | theme_llm
    result = chain.invoke({
        "question": question,
        "conversation": conversation
    })
    
    print("\n=== RAW FOLLOW-UP CLASSIFIER OUTPUT ===")
    print(result.content)
    print("========================================\n")

    response = result.content.strip().upper()
    is_followup = "OUI" in response
    last_chapter = recent_history[0][1] if is_followup else None
    
    return is_followup, last_chapter



chapter_prompt = ChatPromptTemplate.from_messages([
    ("system", """Tu identifies à quel chapitre correspond une question d'élève.

    RÈGLES IMPORTANTES:
    1. Si la question est clairement une suite logique de la conversation récente (pronoms sans antécédent comme "il/elle/ça", questions courtes et vagues comme "Et X?", "Il manque Y?", demandes de clarification), utilise le MÊME chapitre que la conversation récente.
    2. Sinon, identifie le chapitre le plus pertinent selon le contenu de la question.
    3. Si aucun chapitre ne correspond, réponds "Chapitre général".

    Réponds au format:
    CHAPITRE: [Chapitre X: Titre ou Chapitre général]
    RAISON: [Explication courte de ton choix]"""),
                ("human", """Question actuelle: {question}
    {conversation_context}

    Chapitres disponibles: {chapters}

    Quel chapitre?""")
])

