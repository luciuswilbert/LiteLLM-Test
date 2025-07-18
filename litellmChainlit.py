import openai
import fitz  # PyMuPDF
import os
import chainlit as cl
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from dotenv import load_dotenv

load_dotenv()

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

client = openai.OpenAI(api_key="anything", base_url=LITELLM_BASE_URL)

def get_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def chat_completion(messages):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages
    )
    return response.choices[0].message.content

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

class OpenAIEmbeddingsWrapper(Embeddings):
    def embed_documents(self, texts):
        return [get_embedding(text) for text in texts]

    def embed_query(self, text):
        return get_embedding(text)

# Global variable to store the FAISS index
faiss_db = None

@cl.on_chat_start
async def start():
    global faiss_db
    if os.path.exists("my_faiss_index"):
        await cl.Message(
            content="Loading existing knowledge base...",
            author="System"
        ).send()
        embedding_fn = OpenAIEmbeddingsWrapper()
        faiss_db = FAISS.load_local(
            "my_faiss_index",
            embeddings=embedding_fn,
            allow_dangerous_deserialization=True
        )
        await cl.Message(
            content="You can now ask questions about the document.",
            author="System"
        ).send()
    else:
        await cl.Message(
            content="No existing knowledge base found. Please upload a PDF file to create one.",
            author="System"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    global faiss_db
    if message.elements:
        for element in message.elements:
            if hasattr(element, 'name') and element.name.lower().endswith('.pdf'):
                await handle_pdf_upload(element)
            else:
                await cl.Message(
                    content="❌ Please upload a PDF file.",
                    author="System"
                ).send()
        return
    if faiss_db is None:
        await cl.Message(
            content="❌ No knowledge base available. Please upload a PDF file first.",
            author="System"
        ).send()
        return
    user_query = message.content
    try:
        results = faiss_db.similarity_search(user_query, k=4)
        context = "\n\n".join([doc.page_content for doc in results])
        msg = cl.Message(
            content="",
            author="Assistant"
        )
        await msg.send()
        # Compose prompt for LLM
        messages = [
            {"role": "system", "content": "You are AI Assistant. Provide clear, accurate, and concise answers strictly based on the context provided. Ensure your responses are balanced in length—neither too brief nor overly detailed—delivering essential information effectively and efficiently. Avoid including any information not supported by the given context."},
            {"role": "user", "content": f"Context:\n{context}\n\nUser Question: {user_query}\n\nAnswer using only the given context."}
        ]
        answer = chat_completion(messages)
        msg.content = answer if answer is not None else ""
        await msg.update()
    except Exception as e:
        await cl.Message(
            content=f"❌ Error: {str(e)}",
            author="System"
        ).send()

async def handle_pdf_upload(file_element):
    global faiss_db
    try:
        extracted_text = extract_text_from_pdf(file_element.path)
        chunks = chunk_text(extracted_text)
        embedding_fn = OpenAIEmbeddingsWrapper()
        documents = [Document(page_content=chunk) for chunk in chunks]
        faiss_db = FAISS.from_documents(
            documents=documents,
            embedding=embedding_fn
        )
        faiss_db.save_local("my_faiss_index")
        await cl.Message(
            content=f"You can now ask questions about the document.",
            author="System"
        ).send()
    except Exception as e:
        await cl.Message(
            content=f"❌ Error processing file: {str(e)}",
            author="System"
        ).send() 