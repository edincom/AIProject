from langchain_core.runnables import RunnableLambda, RunnableBranch
from app.chains.persona_chain import persona_chain
from app.chains.grading_chain import grading_chain

def rag_lookup(retriever):
    def lookup(inputs):
        if "answer" in inputs:
            inputs["context"] = ""
            return inputs

        query = inputs.get("question", "")
        docs = retriever.invoke(query)
        inputs["context"] = "\n\n".join(d.page_content for d in docs)
        return inputs

    return RunnableLambda(lookup)


def build_router(retriever):
    """Build non-streaming router"""
    rag_chain = rag_lookup(retriever)

    def is_teach_mode(inputs):
        return "answer" not in inputs

    return RunnableBranch(
        (lambda x: is_teach_mode(x), rag_chain | persona_chain),
        grading_chain
    )