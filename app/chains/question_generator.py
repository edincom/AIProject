from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from app.config.settings import LLM_MODEL

llm = ChatMistralAI(model=LLM_MODEL, temperature=1)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un professeur. Génère une question pertinente."),
    ("human", "Instructions: {criteria}\n\nContexte:\n{context}")
])

chain = prompt | llm

def generate_test_question(retriever, criteria):
    docs = retriever.invoke(criteria)
    ctx = "\n\n".join(d.page_content for d in docs)
    return chain.invoke({"criteria": criteria, "context": ctx}).content.strip()
