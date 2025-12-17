from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from app.config.settings import LLM_MODEL
from app.tools.eco_mistral import EcoMistralChat


llm = EcoMistralChat(model=LLM_MODEL, temperature=1)

# Chain for generating test questions (fixed: literal JSON uses doubled braces)
generate_question_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un professeur d'histoire bienveillant mais rigoureux. "
     "À partir du contexte fourni, tu dois générer :\n"
     "1. Une question claire destinée à un élève\n"
     "2. Une réponse attendue complète et correcte. Cette réponse reprend tous les points-clés nécessaires.\n"
     "3. Une liste de points-clés essentiels que l'élève devrait mentionner\n\n"
     "Tu dois absolument retourner UNIQUEMENT un objet JSON, sans texte additionnel."),
    
    ("human",
     "Critères de génération de la question : {criteria}\n\n"
     "Contexte (extraits de documents) :\n{context}\n\n"
     "Format strict du JSON attendu (LES ACCOLADES SONT LITTÉRALES !) :\n"
     "{{\n"
     "  \"question\": \"...\",\n"
     "  \"expected_answer\": \"...\",\n"
     "  \"key_points\": [\"point1\", \"point2\", \"point3\"]\n"
     "}}\n\n"
     "Génère maintenant le JSON.")
])

generate_question_chain = generate_question_prompt | llm


# Chain for grading student answers
test_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un professeur d'histoire rigoureux mais juste qui parle UNIQUEMENT en français. "
     "Tu évalues UNIQUEMENT sur des critères objectifs, pas sur le style. "
     "Tu retournes TOUJOURS un seul objet JSON valide, sans texte supplémentaire."
     "\n\n"
     "LANGUE: Tu dois OBLIGATOIREMENT écrire en français dans TOUS les champs du JSON, "
     "en particulier dans le champ 'advice' qui contient tes conseils pédagogiques."
     "\n\n"
     "RÈGLES JSON CRITIQUES:\n"
     "- Utilise \\n pour les sauts de ligne dans les chaînes, JAMAIS de vrais sauts de ligne\n"
     "- Échappe tous les caractères spéciaux correctement (apostrophes, guillemets, accents)\n"
     "- Toutes les chaînes doivent être sur une seule ligne\n"
     "- Pas de virgules traînantes\n"
     "\n\n"
     "RÈGLES DE NOTATION (total = 100 points):\n"
     "1. Points Clés (50 pts max):\n"
     "   - Compte combien de points clés fournis apparaissent dans la réponse\n"
     "   - Divise 50 par le nombre total de points clés pour obtenir la valeur de chaque point\n"
     "   - Additionne les points pour chaque point clé identifié\n"
     "\n"
     "2. Correspondance Attendue (30 pts max):\n"
     "   - Compare la réponse de l'étudiant avec la réponse attendue\n"
     "   - 30 pts = correspondance excellente (même idées, bien formulées)\n"
     "   - 15-25 pts = correspondance partielle (idées principales présentes)\n"
     "   - 0-10 pts = correspondance faible (manque l'essentiel)\n"
     "\n"
     "3. Faits Incorrects (10 pts max):\n"
     "   - 10 pts = aucune erreur factuelle\n"
     "   - 5 pts = quelques approximations mineures\n"
     "   - 0 pts = erreurs factuelles graves ou nombreuses\n"
     "\n"
     "4. Structure (10 pts max):\n"
     "   - 10 pts = réponse claire, bien organisée, fluide\n"
     "   - 5 pts = structure acceptable mais perfectible\n"
     "   - 0 pts = désorganisé, confus\n"
     "\n"
     "Note finale = Points Clés + Correspondance + Faits Incorrects + Structure"
     ),
    
    ("human",
     "Question posée: {question}\n\n"
     "Réponse de l'étudiant: {answer}\n\n"
     "Réponse attendue (référence): {expected_answer}\n\n"
     "Points clés à identifier: {key_points}\n\n"
     "Retourne UNIQUEMENT un JSON valide (sans sauts de ligne réels dans les chaînes de caractères):\n"
     "{{\n"
     "  \"Question\": \"[Recopie la question]\",\n"
     "  \"Answer\": \"[Recopie la réponse de l'étudiant]\",\n"
     "  \"expected_answer\": \"[Recopie la réponse attendue]\",\n"
     "  \"key_points\": [liste des points clés],\n"
     "  \"scores\": {{\n"
     "      \"Key Points\": [nombre entre 0 et 50],\n"
     "      \"Expected Match\": [nombre entre 0 et 30],\n"
     "      \"Incorrect Facts\": [nombre entre 0 et 10],\n"
     "      \"Structure\": [nombre entre 0 et 10]\n"
     "  }},\n"
     "  \"grade\": [somme totale entre 0 et 100],\n"
     "  \"advice\": \"[Conseils constructifs EN FRANÇAIS pour améliorer la réponse]\"\n"
     "}}\n"
     "\n"
     "RAPPEL IMPORTANT:\n"
     "- Garde toutes les valeurs textuelles sur UNE SEULE LIGNE\n"
     "- Utilise \\n si tu veux indiquer un saut de ligne dans le texte\n"
     "- Le champ 'advice' DOIT contenir des conseils en français, clairs et pédagogiques\n"
     "- N'ajoute AUCUN texte avant ou après le JSON")
])

test_chain = test_prompt | llm
