from fastapi import FastAPI,UploadFile,File
import fitz
from chromadb import Client
from dotenv import load_dotenv
from zai import ZhipuAiClient
from pydantic import BaseModel
import os

from openai import OpenAI
load_dotenv()

deepseek_client = OpenAI(
    api_key=os.getenv("API_KEY"),      # DeepSeek 的 API Key
    base_url="https://api.deepseek.com"
)


zhipu_client = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))

app = FastAPI()
chroma_client = Client()
collection = chroma_client.get_or_create_collection("knowledge_base")

class AskRequest(BaseModel):
    question: str
class AskResponse(BaseModel):
    answer: str
@app.post("/upload-and-index/")
async def upload_and_index(file:UploadFile = File(...)):
    #1.读取字节流
    file_bytes = await file.read()

    #2。从字节流打开PDF并提取文本
    doc = fitz.open(stream=file_bytes,filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()

    #3.分块
    def chunk_text(text, chunk_size=100, overlap=20):
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start:start+chunk_size]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks
    chunks = chunk_text(text)

    # 用智谱 Embedding 模型把每个 chunk 转成向量
    response = zhipu_client.embeddings.create(model="embedding-3", input=chunks)
    embeddings = [item.embedding for item in response.data]   # 提取向量列表
    # 将 chunks 和对应的向量一起存入 Chroma
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]   
         # 自动生成 ID
    )
    return {"success": True, "chunks_count": len(chunks)}
@app.post("/ask/")
async def ask(request: AskRequest):
    query_response = zhipu_client.embeddings.create(
        model="embedding-3",
        input=[request.question]
    )
    query_embedding = query_response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    context = "\n".join(results["documents"][0])
    prompt = f"请根据以下资料回答问题。\n资料：{context}\n问题：{request.question}"
    chat_response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"answer": chat_response.choices[0].message.content}
