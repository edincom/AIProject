from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from app.config.settings import LLM_MODEL

llm = ChatMistralAI(model=LLM_MODEL, temperature=1)

persona_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un professeur bienveillant. Réponds clairement mais sans infantiliser."),
    ("human",
     "Question: {question}\n\nContexte:\n{context}")
])

persona_chain = persona_prompt | llm
