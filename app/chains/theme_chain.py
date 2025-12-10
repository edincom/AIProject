# from dotenv import load_dotenv
# load_dotenv()  # Load API key from .env, only when testing as a single file


from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.tools.loaders import load_pdf
import os


theme_llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0  # Low temperature for structured extraction
)

docs = load_pdf()


def extract_chapters_and_themes(docs, llm):
    """
    Extract chapters and themes from PDF using RAG approach
    """
    
    # Get first 20 pages (usually contains TOC and intro)
    first_pages = docs[:20]
    first_pages_text = "\n\n".join([doc.page_content for doc in first_pages])
    
    # Prompt to extract TOC
    toc_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at analyzing document structure. 
        Extract the table of contents, chapters, and main sections from the document.

        Return a JSON structure like this:
        {{
        "title": "Document Title",
        "chapters": [
            {{
            "number": "1",
            "title": "Chapter Title",
            "page": 10,
            "subsections": ["Subsection 1", "Subsection 2"]
            }}
        ]
        }}

        If no clear chapter structure exists, identify the main sections and themes."""),
                ("user", "Here are the first pages of the document:\n\n{text}\n\nExtract the structure:")
    ])
    
    print("\n🔍 Analyzing document structure...")
    chain = toc_prompt | llm | StrOutputParser()
    toc_result = chain.invoke({"text": first_pages_text[:15000]})  # Limit to avoid token limits
    
    return toc_result



def extract_themes_from_full_doc(docs, llm):
    """
    Extract main themes by analyzing the entire document
    In the case of Atlas.pdf, it is not really necessary to use this function since it already has a clear table of content.
    """
    
    # Sample pages throughout the document
    sample_indices = [0, len(docs)//4, len(docs)//2, 3*len(docs)//4, len(docs)-1]
    sample_pages = [docs[i] for i in sample_indices if i < len(docs)]
    sample_text = "\n\n".join([f"[Page {doc.metadata['page']}]\n{doc.page_content}" for doc in sample_pages])
    
    themes_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at thematic analysis. 
        Analyze the document and identify the main themes, topics, and key concepts.

        Return a JSON structure like this:
        {{
        "main_themes": ["Theme 1", "Theme 2", "Theme 3"],
        "key_topics": ["Topic 1", "Topic 2"],
        "document_type": "textbook/manual/report/etc",
        "summary": "Brief overview of what the document covers"
        }}"""),
                ("user", "Here are sample pages from throughout the document:\n\n{text}\n\nExtract the themes:")
    ])
    
    print("\n🎯 Extracting themes from document...")
    chain = themes_prompt | llm | StrOutputParser()
    themes_result = chain.invoke({"text": sample_text[:15000]})
    
    return themes_result


# Theme extraction + file creation
def themes_and_chapters(docs, llm):
    if not os.path.exists("document_analysis.txt"):
        print("Running theme extraction for the first time...")
        structure = extract_chapters_and_themes(docs, llm)
        themes = extract_themes_from_full_doc(docs, llm)
        
        # Save to file
        with open("document_analysis.txt", "w", encoding="utf-8") as f:
            f.write("DOCUMENT STRUCTURE\n")
            f.write("="*60 + "\n")
            f.write(structure + "\n\n")
            f.write("MAIN THEMES\n")
            f.write("="*60 + "\n")
            f.write(themes + "\n")
        print("💾 Results saved to 'document_analysis.txt'")
        return structure, themes
    else:
        print("Loading cached theme analysis...")
        structure = extract_chapters_and_themes(docs, llm)
        themes = extract_themes_from_full_doc(docs, llm)
        return structure, themes
    


themes_and_chapters(docs, theme_llm)