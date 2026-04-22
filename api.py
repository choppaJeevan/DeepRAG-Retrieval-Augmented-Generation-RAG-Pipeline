import os
import json
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaLLM

from rag_optimized import (
    load_or_process,
    upload_to_weaviate,
    FILE_PATH,
    embed_query,
    rewrite_query,
    weaviate_search,
    rerank_chunks,
    build_prompt,
    GENERATION_MODEL,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    LLM_TEMPERATURE
)

@asynccontextmanager
async def lifespan(app):
    import subprocess, time
    # Start ngrok tunnel on app startup
    ngrok_tunnel = None
    token = os.environ.get("NGROK_AUTHTOKEN", "")
    if token:
        from pyngrok import ngrok
        # Force kill any stale ngrok processes via OS command
        subprocess.run("taskkill /F /IM ngrok.exe", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)  # Wait for OS to fully release the process
        ngrok.set_auth_token(token)
        try:
            ngrok_tunnel = ngrok.connect(8000)
            print(f"\n{'='*60}")
            print(f"  PUBLIC URL (share this!): {ngrok_tunnel.public_url}")
            print(f"  Local URL: http://localhost:8000")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"\nngrok tunnel failed: {e}")
            print("Continuing with local-only mode.\n")
    else:
        print("\nNo NGROK_AUTHTOKEN found - running local only.\n")
    yield
    # Shutdown: cleanly disconnect ngrok
    if ngrok_tunnel:
        from pyngrok import ngrok
        try:
            ngrok.kill()
        except Exception:
            pass

app = FastAPI(title="RAG AI Research Assistant", lifespan=lifespan)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

class ChatRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Save the file temporarily
    temp_path = os.path.join("NLP_project", "temp_upload.pdf")
    os.makedirs("NLP_project", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await file.read())
        
    async def upload_event_stream():
        try:
            yield f"data: {json.dumps({'status': 'Parsing PDF into text chunks using LlamaParse...'})}\n\n"
            
            # Run the heavy, solid-block process in a background thread to prevent blocking
            processed_chunks = await asyncio.to_thread(load_or_process, temp_path)
            
            yield f"data: {json.dumps({'status': f'Vectorizing {len(processed_chunks)} chunks into Weaviate...'})}\n\n"
            await asyncio.to_thread(upload_to_weaviate, processed_chunks)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            yield f"data: {json.dumps({'status': 'Complete!', 'done': True})}\n\n"
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(upload_event_stream(), media_type="text/event-stream")

@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    async def event_stream():
        try:
            # 1. Rewrite query for better retrieval, keep original for reranking/prompt
            rewritten = await asyncio.to_thread(rewrite_query, req.query)
            query_vector = embed_query(rewritten)
            retrieved_chunks = weaviate_search(query_vector, top_k=30)
            top_chunks = rerank_chunks(req.query, retrieved_chunks, top_n=5)
            
            # 2. Build prompt
            prompt = build_prompt(req.query, top_chunks)
            
            # Send context payload first as a custom event so UI can display sources
            sources = [{"page": c.get("page", "Unknown"), "content": c.get("content", "")[:200] + "..." } for c in top_chunks]
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
            
            # 3. Stream LLM output
            llm = OllamaLLM(
                model=GENERATION_MODEL,
                num_ctx=LLM_NUM_CTX,
                num_predict=LLM_NUM_PREDICT,
                temperature=LLM_TEMPERATURE,
            )
            
            in_think_block = False
            for chunk in llm.stream(prompt):
                if "<think>" in chunk:
                    in_think_block = True
                    # Do not yield the literal <think> tag
                    chunk = chunk.replace("<think>", "")
                
                # Check for closing tag
                if "</think>" in chunk:
                    in_think_block = False
                    chunk = chunk.replace("</think>", "")
                    if chunk:
                        yield f"data: {json.dumps({'think': chunk})}\n\n"
                    continue
                    
                if in_think_block and chunk:
                    yield f"data: {json.dumps({'think': chunk})}\n\n"
                elif not in_think_block and chunk:
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                    
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)

