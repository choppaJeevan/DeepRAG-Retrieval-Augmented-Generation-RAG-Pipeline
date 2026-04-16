import os
import re
import time
import json
import hashlib
import weaviate
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

from llama_parse import LlamaParse
from weaviate.classes.config import Configure, Property, DataType
from langchain_core.documents import Document as LC_Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder


from dotenv import load_dotenv
load_dotenv()

FILE_PATH = "./NLP_project/file_survey_paper.pdf"
CACHE_DIR = "./rag_cache"
COLLECTION_NAME = "LlamaParse_MRL_nomic"
TARGET_DIM = 256
EMBED_MODEL_NAME = "nomic-embed-text"
GENERATION_MODEL = "deepseek-r1:8b"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval settings
SEARCH_TOP_K = int(input("How many chunks should I retreive?"))         # Weaviate near-vector retrieval count
RERANK_TOP_N = 5           # Keep top N after re-ranking
MAX_CHUNK_CHARS = 1500     # Truncate chunks before sending to LLM

# Generation settings
LLM_NUM_CTX = 4096         # Context window size
LLM_NUM_PREDICT = 1024     # Max output tokens
LLM_TEMPERATURE = 0.1      # Low = less wandering = faster



# UTILITIES
class Timer:
    """Simple context-manager timer for profiling each step."""
    def __init__(self, label: str):
        self.label = label
    def __enter__(self):
        self.start = time.perf_counter()
        print(f"\n  [{self.label}] Starting...")
        return self
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f" [{self.label}] Done in {self.elapsed:.1f}s")


def file_hash(path: str) -> str:
    """Compute MD5 hash of a file for cache invalidation."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def get_cache_path(pdf_path: str) -> str:
    """Return cache file path based on the PDF filename."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(CACHE_DIR, f"{base}_cache.json")



# STEP 1: PARSE + CHUNK + EMBED (with caching)
def parse_and_chunk(pdf_path: str) -> list[LC_Document]:
    """Parse PDF with LlamaParse → pre-split → semantic chunk."""
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        num_workers=4,        # Increase parallelism for large docs
        verbose=True,
        language="en",
        fast_mode=True,
    )

    print(f"Parsing: {pdf_path}")
    documents = parser.load_data(pdf_path)
    print(f"Parsed {len(documents)} document sections.")

    # Convert LlamaIndex → LangChain documents with metadata
    lc_docs = []
    for doc in documents:
        page_num = doc.metadata.get("page_label", "Unknown")
        lc_docs.append(
            LC_Document(
                page_content=doc.text,
                metadata={
                    "source": doc.metadata.get("file_name", "Unknown"),
                    "page_number": page_num,
                },
            )
        )

    # Pre-split to stay within embedding model token limits
    pre_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
    )
    pre_split_docs = pre_splitter.split_documents(lc_docs)
    print(f"Pre-split into {len(pre_split_docs)} chunks.")

    # Semantic chunking
    embed_model = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    semantic_chunker = SemanticChunker(
        embed_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95.0,
    )
    semantic_chunks = semantic_chunker.split_documents(pre_split_docs)
    print(f"Semantic chunking produced {len(semantic_chunks)} chunks.")

    return semantic_chunks


def batch_embed_and_slice(chunks: list[LC_Document]) -> list[dict]:
    """
    Batch-embed all chunks in a SINGLE call, apply MRL slicing + normalization.
    This replaces TWO separate per-chunk embed_query() loops.
    """
    embed_model = OllamaEmbeddings(model=EMBED_MODEL_NAME)

    # Extract all texts for batch embedding
    all_texts = [chunk.page_content for chunk in chunks]

    print(f"Batch-embedding {len(all_texts)} chunks with {EMBED_MODEL_NAME}...")
    all_vectors = embed_model.embed_documents(all_texts)  # Single batched call!

    # MRL slicing + normalization (vectorized with NumPy)
    vectors_np = np.array(all_vectors, dtype=np.float32)[:, :TARGET_DIM]
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    vectors_np = vectors_np / norms

    print(f"MRL: Sliced to {TARGET_DIM} dims and normalized.")

    # Build processed chunks with metadata
    processed = []
    for i, chunk in enumerate(chunks):
        processed.append({
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "vector": vectors_np[i].tolist(),
        })

    return processed


def load_or_process(pdf_path: str) -> list[dict]:
    """
    Main caching layer. Returns processed chunks with vectors.
    If cache exists and PDF hasn't changed, loads from cache.
    Otherwise, parses → chunks → embeds → caches.
    """
    cache_path = get_cache_path(pdf_path)
    current_hash = file_hash(pdf_path)

    # Check cache
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        if cache_data.get("file_hash") == current_hash:
            chunks = cache_data["chunks"]
            print(f"Cache hit! Loaded {len(chunks)} pre-computed chunks.")
            return chunks
        else:
            print("PDF changed — re-processing...")

    # Cache miss: full pipeline
    with Timer("Parse + Chunk"):
        semantic_chunks = parse_and_chunk(pdf_path)

    with Timer("Batch Embed + MRL"):
        processed_chunks = batch_embed_and_slice(semantic_chunks)

    # Save to cache (with hash for invalidation)
    cache_data = {
        "file_hash": current_hash,
        "file_path": pdf_path,
        "target_dim": TARGET_DIM,
        "num_chunks": len(processed_chunks),
        "chunks": processed_chunks,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False)
    print(f"Cached {len(processed_chunks)} chunks to {cache_path}")

    return processed_chunks

# STEP 2: WEAVIATE UPLOAD (using pre-computed vectors)
def upload_to_weaviate(processed_chunks: list[dict]):
    """Upload chunks with pre-computed MRL vectors. No re-embedding!"""
    client = weaviate.connect_to_local()

    try:
        # Re-create collection for fresh schema
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)

        client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="page_number", data_type=DataType.TEXT),
            ],
        )

        collection = client.collections.get(COLLECTION_NAME)

        # Batch upload with PRE-COMPUTED vectors (no embed_query calls!)
        with collection.batch.dynamic() as batch:
            for item in processed_chunks:
                page_val = str(
                    item["metadata"].get(
                        "page_label",
                        item["metadata"].get("page_number", "Unknown"),
                    )
                )
                batch.add_object(
                    properties={
                        "content": item["content"],
                        "source": str(item["metadata"].get("source", "Unknown")),
                        "page_number": page_val,
                    },
                    vector=item["vector"],
                )

        print(f"Uploaded {len(processed_chunks)} chunks to Weaviate ({COLLECTION_NAME}).")

    finally:
        client.close()


# STEP 3: QUERY EMBEDDING
def embed_query(query_text: str) -> list[float]:
    """Embed a user query, apply MRL slicing + normalization."""
    embed_model = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    full_vector = embed_model.embed_query(query_text)

    # MRL slice + normalize
    sliced = np.array(full_vector[:TARGET_DIM], dtype=np.float32)
    norm = np.linalg.norm(sliced)
    if norm > 0:
        sliced = sliced / norm

    return sliced.tolist()


# STEP 4: WEAVIATE VECTOR SEARCH
def weaviate_search(query_vec: list[float], top_k: int = SEARCH_TOP_K) -> list[dict]:
    """
    Search Weaviate using near_vector with pre-computed MRL vectors.
    Weaviate uses HNSW internally — O(log n) search, scales to millions.
    """
    client = weaviate.connect_to_local()

    try:
        collection = client.collections.get(COLLECTION_NAME)

        response = collection.query.near_vector(
            near_vector=query_vec,
            limit=top_k,
            return_properties=["content", "page_number", "source"],
        )

        results = []
        for obj in response.objects:
            results.append({
                "content": obj.properties["content"],
                "page": obj.properties.get("page_number", "Unknown"),
                "source": obj.properties.get("source", "Unknown"),
            })

        print(f"Weaviate returned {len(results)} results.")
        return results

    finally:
        client.close()


# STEP 5: CROSS-ENCODER RE-RANKING (no LLM distillation!)
def rerank_chunks(query: str, chunks: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Re-rank retrieved chunks using a cross-encoder.
    This REPLACES the old LLM distillation step (gemma3:4b) entirely.
    Cross-encoders are purpose-built for relevance scoring — faster + better.
    """
    reranker = CrossEncoder(RERANKER_MODEL)

    # Prepare query-document pairs
    pairs = [[query, chunk["content"]] for chunk in chunks]

    # Score all pairs
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for i, score in enumerate(scores):
        chunks[i]["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    print(f"Re-ranked {len(chunks)} → keeping top {top_n}")
    for i, c in enumerate(ranked[:top_n]):
        print(f"   Rank {i+1}: Page {c['page']} | Score: {c['rerank_score']:.4f}")

    return ranked[:top_n]


# STEP 6: PROMPT SYNTHESIS
def build_prompt(query: str, top_chunks: list[dict]) -> str:
    """
    Build a grounded prompt with truncated context.
    MAX_CHUNK_CHARS prevents context explosion for large documents.
    """
    system_rules = (
        "You are a professional research assistant. "
        "Use ONLY the provided context to answer the question.\n"
        "If the answer is not in the context, say you don't know.\n"
        "For every fact, cite the page number as [Page X].\n"
        "Be concise and direct.\n\n"
        "CONSTRAINTS:\n"
        "- DO NOT use any outside knowledge or training data. Only use the CONTEXT below.\n"
        "- DO NOT speculate, assume, or infer beyond what is explicitly stated.\n"
        "- DO NOT repeat the question or paraphrase it back.\n"
        "- DO NOT add disclaimers like 'based on the context provided'.\n"
        "- DO NOT produce long chain-of-thought reasoning. Give the answer directly.\n"
        "- If multiple pages support a fact, cite all of them."
    )

    context_block = ""
    for i, chunk in enumerate(top_chunks):
        # Truncate each chunk to prevent context explosion
        content = chunk["content"][:MAX_CHUNK_CHARS]
        context_block += f"\n--- CHUNK {i+1} (PAGE {chunk['page']}) ---\n{content}\n"

    prompt = f"""{system_rules}

CONTEXT:
{context_block}

USER QUESTION:
{query}

GROUNDED ANSWER:"""

    return prompt



# STEP 7: LLM GENERATION
def generate_answer(prompt: str) -> str:
    """Generate a grounded answer using DeepSeek-R1:8b with capped parameters."""
    llm = OllamaLLM(
        model=GENERATION_MODEL,
        num_ctx=LLM_NUM_CTX,
        num_predict=LLM_NUM_PREDICT,
        temperature=LLM_TEMPERATURE,
    )

    print(f"{GENERATION_MODEL} is generating...")
    raw_response = llm.invoke(prompt)

    # DeepSeek-R1 wraps reasoning in <think>...</think> tags.
    # Strip them to get only the visible answer.
    answer = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

    if not answer:
        # If stripping left nothing, the model only produced thinking — return raw
        print("[WARNING] Model produced only <think> content. Returning raw response.")
        return raw_response

    return answer



# MAIN PIPELINE
def main():
    timings = {}
    total_start = time.perf_counter()

    # ── Step 1: Parse + Embed (cached) ──
    with Timer("Parse + Embed (cached)") as t:
        processed_chunks = load_or_process(FILE_PATH)
    timings["1. Parse + Embed"] = t.elapsed

    # ── Step 2: Upload to Weaviate ──
    with Timer("Weaviate Upload") as t:
        upload_to_weaviate(processed_chunks)
    timings["2. Weaviate Upload"] = t.elapsed

    # ── Step 3: User Query + Embedding ──
    query_text = input("\nEnter your question: ")

    with Timer("Query Embedding") as t:
        query_vector = embed_query(query_text)
    timings["3. Query Embed"] = t.elapsed

    # ── Step 4: Weaviate Search ──
    with Timer("Weaviate Search") as t:
        retrieved_chunks = weaviate_search(query_vector, top_k=SEARCH_TOP_K)
    timings["4. Weaviate Search"] = t.elapsed

    # ── Step 5: Re-rank ──
    with Timer("Cross-Encoder Re-ranking") as t:
        top_chunks = rerank_chunks(query_text, retrieved_chunks, top_n=RERANK_TOP_N)
    timings["5. Re-ranking"] = t.elapsed

    # ── Step 6: Prompt Synthesis ──
    with Timer("Prompt Synthesis") as t:
        final_prompt = build_prompt(query_text, top_chunks)
    timings["6. Prompt Synthesis"] = t.elapsed

    # ── Step 7: LLM Generation ──
    with Timer(f"LLM Generation ({GENERATION_MODEL})") as t:
        answer = generate_answer(final_prompt)
    timings["7. LLM Generation"] = t.elapsed

    # ── Output ──
    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 60)
    print("FINAL RESPONSE:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

    # Performance summary
    print("\n PERFORMANCE SUMMARY:")
    print("-" * 40)
    for step, elapsed in timings.items():
        bar = "█" * int(elapsed / total_elapsed * 30)
        print(f"  {step:<25} {elapsed:>6.1f}s  {bar}")
    print("-" * 40)
    print(f"  {'TOTAL':<25} {total_elapsed:>6.1f}s")
    print()


if __name__ == "__main__":
    main()