import fitz
import base64
from langchain_community.docstore.document import Document
from langchain_core.messages import HumanMessage
from app.config.settings import PDF_PATH, VISION_MODEL
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyMuPDFLoader

vision_llm = ChatMistralAI(model=VISION_MODEL, temperature=0)

def caption_images():
    pass
    # pdf = fitz.open(PDF_PATH)
    # image_docs = []

    # for page_num in range(len(pdf)):
    #     page = pdf.load_page(page_num)

    #     for img_index, info in enumerate(page.get_images(full=True)):
    #         xref = info[0]
    #         img = pdf.extract_image(xref)

    #         b64 = base64.b64encode(img["image"]).decode("utf-8")
    #         mime = {
    #             "png": "image/png",
    #             "jpg": "image/jpeg",
    #             "jpeg": "image/jpeg"
    #         }.get(img.get("ext", "png"), "image/png")

    #         msg = HumanMessage(content=[
    #             {"type": "text", "text": "Describe this image in natural language."},
    #             {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    #         ])

    #         caption = vision_llm.invoke([msg]).content

    #         image_docs.append(
    #             Document(
    #                 page_content=caption,
    #                 metadata={
    #                     "type": "image_caption",
    #                     "page": page_num + 1,
    #                     "image_index": img_index,
    #                     "source": PDF_PATH
    #                 }
    #             )
    #         )

    # pdf.close()
    # return image_docs



def load_pdf():
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")
    return docs
