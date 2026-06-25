import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

if not os.getenv("QDRANT_API_KEY"):
    raise ValueError("QDRANT_API_KEY not found")

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found")

app = FastAPI(title="Reliance Policy")

embedding_model = SentenceTransformer("BAAI/bge-small-en")

qdrant_client = QdrantClient(
    url="https://1cd825a0-0f3f-49f4-b4aa-0d3dc305290d.australia-southeast1-0.gcp.cloud.qdrant.io",
    api_key= os.getenv("QDRANT_API_KEY")
)


llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key= os.getenv("GROQ_API_KEY")
)

class QuestionRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/search")
async def search_documents(request: QuestionRequest):

    try:
        query_vector = embedding_model.encode(request.query).tolist()

        search_results = qdrant_client.query_points(
        collection_name="rag_docs",
        query=query_vector,          
        limit=request.top_k
        ).points
        
        retrieved_texts = [hit.payload.get("text", "") for hit in search_results]
        
        return {
            "query": request.query, 
            "results": retrieved_texts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):

    try:

        query_vector = embedding_model.encode(request.query).tolist()

        search_results = qdrant_client.search(
            collection_name="rag_docs",
            query_vector=query_vector,
            limit=request.top_k
        )
        
        contexts = [hit.payload.get("text", "") for hit in search_results]

        context_string = "\n\n---\n\n".join(contexts)

        prompt = (
            "You are an AI assistant for Reliance Policy. "
            "Use ONLY the following context to answer the user's question. "
            "If the answer is not in the context, say 'I don't have enough "
            "information in the policy documents to answer.'\n\n"
            f"CONTEXT:\n{context_string}"
        )

        response = llm_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": request.query}
            ],
            temperature=0.2 
        )
        
        return {
            "query": request.query,
            "answer": response.choices[0].message.content,
            "source_contexts": contexts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))