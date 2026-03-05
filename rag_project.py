

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file_path: str) -> str:


    reader = PdfReader(file_path)
    
    txt = ''
    for i in reader.pages:
        txt = txt + i.extract_text()
    
    return txt


def chunk_text(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)


text = extract_text_from_pdf(r'C:\rag\file-sample_150kB.pdf')
chunks = chunk_text(text)

print(chunks)