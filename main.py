from fastapi import FastAPI, UploadFile, File
from models import QueryReq, QueryResp
from rag import extract_text_from_pdf,chunk_text,embed_chunks,store_in_qdrant, query
import shutil
import tempfile


app = FastAPI()

@app.post("/upload/")
def uploadFile(file:UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    text = extract_text_from_pdf(tmp_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    store_in_qdrant(chunks,embeddings)
    return {"message": "file uploaded and store in vector store"}

@app.post("/query/")
def query_endpoint(req:QueryReq):
    ai_response = query(req.query,req.top_k)
    return QueryResp(answer = ai_response)