
import os
import uuid
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams ,PointStruct


load_dotenv()


client = OpenAI()
embedding_model = os.getenv("EMBEDDING_MODEL") 
qdrant_client = QdrantClient(url="http://localhost:6335")

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
    response = client.embeddings.create(input=chunks, model=embedding_model)
    return [embedding.embedding for embedding in response.data]

#docker run -p 6333:6333 qdrant/qdrant
def store_in_qdrant(chunks: list[str], embeddings: list[list[float]], file_name:str) -> None:
    
    if not qdrant_client.collection_exists(collection_name="docs"):
        qdrant_client.create_collection(
            collection_name="docs",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    for chunk,embedding in zip(chunks,embeddings):
        qdrant_client.upsert(
            collection_name="docs",
            points=[
                PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"text":chunk, "source":file_name})
            ]
        )

    result = qdrant_client.get_collection(collection_name="docs")
    return result.points_count

def query(query:str,top_k:int = 2)->str:
    
    embedd_query = client.embeddings.create(input= query, model=embedding_model)
    
    
    response = qdrant_client.query_points(
        collection_name='docs',
        query = embedd_query.data[0].embedding,
        limit = top_k,
    )

    context = [point.payload.get("text") for point in response.points]

    context_str = "\n\n".join(context)

    ai_response = client.chat.completions.create(
        model = "o4-mini-2025-04-16",
        messages = [
            {"role" : "system","content" : f"answer the question based on  the following context{context_str}"},
            {"role" : "user","content" : query},
        ],
    )

    return ai_response.choices[0].message.content


# text = extract_text_from_pdf(r'C:\rag\CIVIL SERVANTS LEAVE RULES, 1986.pdf')
# chunks = chunk_text(text)
# embeddings = embed_chunks(chunks)
# store_in_qdrant(chunks, embeddings)
# ai = query("what policy of recreation leave")
# print(ai)