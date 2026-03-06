from pydantic import BaseModel
class QueryReq(BaseModel):
    query: str
    top_k:int = 2

class QueryResp(BaseModel):
    answer:str

