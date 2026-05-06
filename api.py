import os
import sys
import json
import asyncio
import threading
import signal
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
    LLM_TEMPERATURE,
    COLLECTION_NAME
)

# ── Playwright: auto-launch App Window ──────────────────────────────────
_playwright_instance = None
_browser_instance = None
_browser_context = None

def launch_browser(url: str):
    """Launch an isolated browser window via Playwright."""
    global _playwright_instance, _browser_instance, _browser_context

    def _run():
        global _playwright_instance, _browser_instance, _browser_context
        import time
        time.sleep(1.5) # Wait for server to start
        try:
            from playwright.sync_api import sync_playwright
            _playwright_instance = sync_playwright().start()
            
            # Try to find an installed browser in this order
            channels = ["msedge","chrome", None]
            
            for channel in channels:
                try:
                    kwargs = {"headless": False}
                    if channel:
                        kwargs["channel"] = channel
                    _browser_instance = _playwright_instance.chromium.launch(**kwargs)
                    break
                except Exception:
                    pass
            
            # If Chromium failed, try WebKit (Safari on Mac)
            if not _browser_instance:
                try:
                    _browser_instance = _playwright_instance.webkit.launch(headless=False)
                except Exception:
                    pass
            
            if not _browser_instance:
                raise Exception("No Playwright browsers found.")

            _browser_context = _browser_instance.new_context()
            page = _browser_context.new_page()
            page.goto(url)
            print(f"  Isolated browser opened at: {url}")
            
        except Exception as e:
            print(f"\n  [WARNING] Playwright isolated window failed.")
            print(f"  Falling back to system browser...")
            import webbrowser
            webbrowser.open_new(url)

    import threading
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

def close_browser():
    """Cleanly shut down Playwright browser and instance."""
    global _playwright_instance, _browser_instance, _browser_context
    try:
        if _browser_context:
            _browser_context.close()
        if _browser_instance:
            _browser_instance.close()
        if _playwright_instance:
            _playwright_instance.stop()
    except Exception:
        pass
    _browser_context = None
    _browser_instance = None
    _playwright_instance = None


# ── Clear Weaviate on startup so previous sessions don't bleed through ──────
def clear_weaviate_collection():
    """Delete the Weaviate collection if it exists to start fresh."""
    try:
        import weaviate
        client = weaviate.connect_to_local()
        try:
            if client.collections.exists(COLLECTION_NAME):
                client.collections.delete(COLLECTION_NAME)
                print(f"  Cleared previous Weaviate collection: {COLLECTION_NAME}")
            else:
                print(f"  No existing Weaviate collection to clear.")
        finally:
            client.close()
    except Exception as e:
        print(f"  [WARNING] Could not clear Weaviate: {e}")


# ── FastAPI lifespan ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    import subprocess, time

    # 1. Clear previous session data from Weaviate
    print("\n  Cleaning up previous session data...")
    clear_weaviate_collection()

    # 2. Start ngrok tunnel (optional, for sharing)
    ngrok_tunnel = None
    token = os.environ.get("NGROK_AUTHTOKEN", "")
    share_url = "http://localhost:8000"

    if token:
        from pyngrok import ngrok
        # Force kill any stale ngrok processes via OS command
        subprocess.run("taskkill /F /IM ngrok.exe", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)  # Wait for OS to fully release the process
        ngrok.set_auth_token(token)
        try:
            ngrok_tunnel = ngrok.connect(8000)
            share_url = ngrok_tunnel.public_url
            print(f"\n{'='*60}")
            print(f"  PUBLIC URL (share this!): {share_url}")
            print(f"  Local URL: http://localhost:8000")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"\nngrok tunnel failed: {e}")
            print("Continuing with local-only mode.\n")
    else:
        print(f"\n{'='*60}")
        print(f"  Local URL: http://localhost:8000")
        print(f"{'='*60}\n")

    # 3. Auto-launch the default web browser
    launch_browser("http://localhost:8000")

    yield

    # Shutdown: cleanly disconnect everything
    print("\n  Shutting down...")
    close_browser()
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

@app.post("/api/reset")
async def reset_session():
    """Clear all session data: Weaviate collection + signal frontend to clear."""
    clear_weaviate_collection()
    return JSONResponse({"status": "Session cleared"})

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
    """Stream pipeline status updates and LLM tokens via SSE."""

    async def event_stream():
        import queue as queue_mod
        import time

        msg_queue = queue_mod.Queue()

        def _full_pipeline():
            """Run the entire RAG pipeline in a single thread, pushing events to the queue."""
            try:
                # ── Step 1: Embed query ──
                msg_queue.put(("status", {"step": 1, "total": 4, "label": "Embedding your query", "state": "active"}))
                t0 = time.perf_counter()
                query_vector = embed_query(req.query)
                elapsed = time.perf_counter() - t0
                msg_queue.put(("status", {"step": 1, "total": 4, "label": "Query embedded", "state": "done", "time": f"{elapsed:.1f}s"}))
                print(f"  [CHAT] Embedding done ({elapsed:.1f}s)")

                # ── Step 2: Vector search ──
                msg_queue.put(("status", {"step": 2, "total": 4, "label": "Searching document chunks", "state": "active"}))
                t1 = time.perf_counter()
                retrieved_chunks = weaviate_search(query_vector, top_k=30)
                elapsed = time.perf_counter() - t1
                msg_queue.put(("status", {"step": 2, "total": 4, "label": f"Found {len(retrieved_chunks)} relevant chunks", "state": "done", "time": f"{elapsed:.1f}s"}))
                print(f"  [CHAT] Search done ({elapsed:.1f}s)")

                # ── Step 3: Re-rank ──
                msg_queue.put(("status", {"step": 3, "total": 4, "label": f"Re-ranking {len(retrieved_chunks)} chunks", "state": "active"}))
                t2 = time.perf_counter()
                top_chunks = rerank_chunks(req.query, retrieved_chunks, top_n=5)
                elapsed = time.perf_counter() - t2
                msg_queue.put(("status", {"step": 3, "total": 4, "label": f"Top {len(top_chunks)} chunks selected", "state": "done", "time": f"{elapsed:.1f}s"}))
                print(f"  [CHAT] Reranking done ({elapsed:.1f}s)")

                # ── Step 4: Generate ──
                prompt = build_prompt(req.query, top_chunks)
                msg_queue.put(("status", {"step": 4, "total": 4, "label": "Generating answer", "state": "active"}))

                # Send sources
                sources = [{"page": c.get("page", "Unknown"), "content": c.get("content", "")[:200] + "..."} for c in top_chunks]
                msg_queue.put(("sources", sources))

                # Stream LLM
                llm = OllamaLLM(
                    model=GENERATION_MODEL,
                    num_ctx=LLM_NUM_CTX,
                    num_predict=LLM_NUM_PREDICT,
                    temperature=LLM_TEMPERATURE,
                )
                print(f"  [CHAT] LLM streaming started...")
                for tok in llm.stream(prompt):
                    msg_queue.put(("token", tok))

                msg_queue.put(("status", {"step": 4, "total": 4, "label": "Response complete", "state": "done"}))
                msg_queue.put(("done", None))
                print(f"  [CHAT] Response complete.")

            except Exception as e:
                print(f"  [CHAT] ERROR: {e}")
                msg_queue.put(("error", str(e)))

        # Start the entire pipeline in a single background thread
        pipeline_thread = threading.Thread(target=_full_pipeline, daemon=True)
        pipeline_thread.start()

        # Read events from the queue and yield SSE
        in_think_block = False
        while True:
            msg = await asyncio.to_thread(msg_queue.get)
            msg_type, msg_data = msg

            if msg_type == "status":
                yield f"event: status\ndata: {json.dumps(msg_data)}\n\n"

            elif msg_type == "sources":
                yield f"event: sources\ndata: {json.dumps(msg_data)}\n\n"

            elif msg_type == "token":
                chunk = msg_data
                if "<think>" in chunk:
                    in_think_block = True
                    chunk = chunk.replace("<think>", "")

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

            elif msg_type == "done":
                yield "data: [DONE]\n\n"
                break

            elif msg_type == "error":
                yield f"event: error\ndata: {json.dumps({'error': msg_data})}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
