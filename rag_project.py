
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(size=14, distance=Distance.DOT),
)


load_dotenv()

def extract_text_from_pdf(file_path: str) -> str:


    reader = PdfReader(file_path)
    
    txt = ''
    for i in reader.pages:
        txt = txt + i.extract_text()
    
    return txt

def chunk_text(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)

def embed_chunks(chunks:list[str])->list[list[float]]:
    client = OpenAI()
    embedding_model = os.getenv("EMBEDDING_MODEL")
    response = client.embeddings.create(input=chunks, model=embedding_model)
    return [embedding.embedding for embedding in response.data]

#docker run -p 6333:6333 qdrant/qdrant
def store_in_qdrant(chunks: list[str], embeddings: list[list[float]]) -> None:
    for chunck,embedding in zip(chunks,embeddings):
        pass


text = extract_text_from_pdf(r'C:\rag\file-sample_150kB.pdf')
chunks = chunk_text(text)
embeddings = embed_chunks(chunks)

print(f"Total embeddings: {len(embeddings)}")
print(f"Vector dimensions: {len(embeddings[0])}")