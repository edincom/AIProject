from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from app.config.settings import LLM_MODEL

llm = ChatMistralAI(model=LLM_MODEL, temperature=0)

grading_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un correcteur automatique strict. Retourne uniquement du JSON."),
    ("human",
     "Barème: {rubric}\n"
     "Question: {question}\n"
     "Réponse: {answer}")
])

grading_chain = grading_prompt | llm
