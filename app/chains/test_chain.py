from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from app.config.settings import LLM_MODEL

llm = ChatMistralAI(model=LLM_MODEL, temperature=0)

# Chain for generating test questions
generate_question_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Tu es un professeur bienveillant et rigoureux. "
     "À partir du contexte fourni, génère une question pertinente pour un élève."),
    ("human",
     "Instructions : {criteria}\n\nContexte : {context}")
])

generate_question_chain = generate_question_prompt | llm

# Chain for grading student answers
test_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Act as a supportive but rigorous history teacher.\n"
     "Your goal is to evaluate the student's answer and return ONLY a JSON object."),
    ("human",
     "Grading rubric: {grading_rubric}\n"
     "Question: {question}\n"
     "Answer: {answer}\n"
     "Scores template: {scores_text}\n"
     "Constraints: grade MUST equal sum of all scores.\n"
     "Return a JSON object with keys:\n"
     "- Section (the general theme)\n"
     "- Question\n"
     "- Answer\n"
     "- grade (0-100)\n"
     "- scores (object with Pertinence, Faits non corrects, Faits manquants, Structure)\n"
     "- advice (string with improvement suggestions)\n"
     "No extra text or Markdown, ONLY JSON.")
])

test_chain = test_prompt | llm