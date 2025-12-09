from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from app.config.settings import LLM_MODEL

# Non-streaming LLM
llm = ChatMistralAI(model=LLM_MODEL, temperature=1)

# Streaming LLM - same model but for streaming responses
streaming_llm = ChatMistralAI(model=LLM_MODEL, temperature=1, streaming=True)

persona_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un professeur bienveillant. Réponds clairement mais sans infantiliser."),
    ("human",
     "Question: {question}\n\nContexte:\n{context}")
])

# Non-streaming chain
persona_chain = persona_prompt | llm

# Streaming chain
streaming_persona_chain = persona_prompt | streaming_llm