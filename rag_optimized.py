import os
import re
import time
import json
import random
import difflib
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
SEARCH_TOP_K = 30              # Default; overridden in main() via input
RERANK_TOP_N = 5           # Keep top N after re-ranking
MAX_CHUNK_CHARS = 1500     # Truncate chunks before sending to LLM

# Generation settings
LLM_NUM_CTX = 4096         # Context window size
LLM_NUM_PREDICT = 1024     # Max output tokens
LLM_TEMPERATURE = 0.1      # Low = less wandering = faster

# Evaluation settings
EVAL_DATASET_SIZE = 20
EVAL_RANDOM_SEED = 42
EVAL_MAX_FAILURE_EXAMPLES = 5
EVAL_DATASET_PATH = os.path.join(CACHE_DIR, "evaluation_dataset.json")
EVAL_RESULTS_PATH = os.path.join(CACHE_DIR, "evaluation_results.json")



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


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize text for lightweight lexical metrics."""
    return re.findall(r"\b\w+\b", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    ref_counts = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    overlap = 0
    for tok, cnt in pred_counts.items():
        overlap += min(cnt, ref_counts.get(tok, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:])) if len(tokens) > 1 else set()


def grounded_bigram_recall(answer: str, context: str) -> float:
    """Proxy groundedness: answer bigrams also present in retrieved context."""
    answer_bigrams = bigrams(tokenize(answer))
    if not answer_bigrams:
        return 0.0
    context_bigrams = bigrams(tokenize(context))
    overlap = len(answer_bigrams.intersection(context_bigrams))
    return overlap / len(answer_bigrams)


def extract_json_object(text: str) -> dict | None:
    """Attempt to parse the first JSON object in LLM output."""
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None



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
    )

    print(f"Parsing: {pdf_path}")
    documents = parser.load_data(pdf_path)
    print(f"Parsed {len(documents)} document sections.")

    # Convert LlamaIndex → LangChain documents with metadata
    # NOTE: LlamaParse returns empty metadata (no page_label).
    # Since it returns ~1 section per page, we use the index as page number.
    lc_docs = []
    for idx, doc in enumerate(documents):
        page_num = doc.metadata.get("page_label", str(idx + 1))
        lc_docs.append(
            LC_Document(
                page_content=doc.text,
                metadata={
                    "source": doc.metadata.get("file_name", os.path.basename(pdf_path)),
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


def build_synthetic_qa_prompt(chunk_text: str, page_number: str) -> str:
    """Prompt template for generating synthetic grounded Q/A."""
    return f"""You are generating evaluation data for a RAG system.
Create one grounded question-answer pair from the context below.

Rules:
- The question must be answerable from this context alone.
- The answer must be concise (1-3 sentences), factual, and directly grounded.
- Include page citation in answer exactly as [Page {page_number}].
- Return ONLY valid JSON with keys: question, answer.

CONTEXT:
{chunk_text}
"""


def generate_synthetic_sample(chunk: dict, chunk_id: int) -> dict | None:
    """Generate one synthetic Q/A sample from a chunk."""
    page = str(chunk["metadata"].get("page_number", "Unknown"))
    content = chunk["content"][:MAX_CHUNK_CHARS]
    prompt = build_synthetic_qa_prompt(content, page)
    raw = generate_answer(prompt)
    parsed = extract_json_object(raw)
    if not parsed:
        return None

    question = str(parsed.get("question", "")).strip()
    answer = str(parsed.get("answer", "")).strip()
    if len(question) < 12 or len(answer) < 20:
        return None
    if "[Page" not in answer:
        return None

    return {
        "id": f"synth_{chunk_id}",
        "question": question,
        "reference_answer": answer,
        "source_page": page,
        "source_chunk_id": chunk_id,
    }


def is_near_duplicate(question: str, existing_questions: list[str], threshold: float = 0.9) -> bool:
    """Simple similarity check to avoid duplicate synthetic questions."""
    q_norm = " ".join(tokenize(question))
    for existing in existing_questions:
        e_norm = " ".join(tokenize(existing))
        if not q_norm or not e_norm:
            continue
        ratio = difflib.SequenceMatcher(None, q_norm, e_norm).ratio()
        if ratio >= threshold:
            return True
    return False


def save_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_cached_eval_dataset(expected_hash: str, requested_size: int) -> list[dict] | None:
    if not os.path.exists(EVAL_DATASET_PATH):
        return None
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("file_hash") != expected_hash:
        return None
    samples = data.get("samples", [])
    if len(samples) < requested_size:
        return None
    return samples[:requested_size]


def build_synthetic_dataset(processed_chunks: list[dict], file_hash_value: str, dataset_size: int = EVAL_DATASET_SIZE) -> list[dict]:
    """Generate or load cached synthetic Q/A dataset."""
    cached_samples = load_cached_eval_dataset(file_hash_value, dataset_size)
    if cached_samples is not None:
        print(f"Loaded cached synthetic eval dataset with {len(cached_samples)} samples.")
        return cached_samples

    random.seed(EVAL_RANDOM_SEED)
    sampled_indices = list(range(len(processed_chunks)))
    random.shuffle(sampled_indices)

    samples = []
    existing_questions = []
    max_attempts = min(len(sampled_indices), dataset_size * 3)
    for idx in sampled_indices[:max_attempts]:
        sample = generate_synthetic_sample(processed_chunks[idx], idx)
        if not sample:
            continue
        if is_near_duplicate(sample["question"], existing_questions):
            continue
        samples.append(sample)
        existing_questions.append(sample["question"])
        print(f"Synthetic sample {len(samples)}/{dataset_size} generated.")
        if len(samples) >= dataset_size:
            break

    if not samples:
        raise RuntimeError("Failed to generate any synthetic evaluation samples.")

    dataset_payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_hash": file_hash_value,
        "file_path": FILE_PATH,
        "dataset_size": len(samples),
        "samples": samples,
    }
    save_json(EVAL_DATASET_PATH, dataset_payload)
    print(f"Saved synthetic evaluation dataset to {EVAL_DATASET_PATH}")
    return samples


def evaluate_one_sample(sample: dict, top_k: int) -> dict:
    """Run full retrieval + generation pipeline for one evaluation sample."""
    step_times = {}

    with Timer("Eval: Query Embedding") as t:
        query_vector = embed_query(sample["question"])
    step_times["query_embed_s"] = t.elapsed

    with Timer("Eval: Retrieval") as t:
        retrieved_chunks = weaviate_search(query_vector, top_k=top_k)
    step_times["retrieval_s"] = t.elapsed

    with Timer("Eval: Re-ranking") as t:
        top_chunks = rerank_chunks(sample["question"], retrieved_chunks, top_n=RERANK_TOP_N)
    step_times["rerank_s"] = t.elapsed

    with Timer("Eval: Prompt Build") as t:
        final_prompt = build_prompt(sample["question"], top_chunks)
    step_times["prompt_s"] = t.elapsed

    with Timer("Eval: Answer Generation") as t:
        answer = generate_answer(final_prompt)
    step_times["generation_s"] = t.elapsed

    retrieved_pages_before = [str(c.get("page", "Unknown")) for c in retrieved_chunks]
    retrieved_pages_after = [str(c.get("page", "Unknown")) for c in top_chunks]
    source_page = str(sample["source_page"])

    rr = 0.0
    if source_page in retrieved_pages_before:
        rank = retrieved_pages_before.index(source_page) + 1
        rr = 1.0 / rank

    combined_context = "\n".join(chunk["content"] for chunk in top_chunks)
    metrics = {
        "hit_at_k_before": int(source_page in retrieved_pages_before),
        "hit_at_n_after": int(source_page in retrieved_pages_after),
        "reciprocal_rank": rr,
        "token_f1": token_f1(answer, sample["reference_answer"]),
        "citation_ok": int(bool(re.search(r"\[Page\s+\d+\]", answer))),
        "grounded_bigram_recall": grounded_bigram_recall(answer, combined_context),
    }

    return {
        "sample_id": sample["id"],
        "question": sample["question"],
        "reference_answer": sample["reference_answer"],
        "source_page": source_page,
        "retrieved_pages_before": retrieved_pages_before,
        "retrieved_pages_after": retrieved_pages_after,
        "generated_answer": answer,
        "metrics": metrics,
        "timings": step_times,
    }


def summarize_eval_results(results: list[dict]) -> dict:
    """Aggregate retrieval and generation quality metrics."""
    if not results:
        return {}

    count = len(results)
    metric_keys = [
        "hit_at_k_before",
        "hit_at_n_after",
        "reciprocal_rank",
        "token_f1",
        "citation_ok",
        "grounded_bigram_recall",
    ]

    aggregates = {}
    for key in metric_keys:
        aggregates[key] = float(np.mean([r["metrics"][key] for r in results]))

    reranker_lift = aggregates["hit_at_n_after"] - aggregates["hit_at_k_before"]
    avg_total_latency = float(
        np.mean(
            [
                sum(r["timings"].values())
                for r in results
            ]
        )
    )

    failures = sorted(
        results,
        key=lambda r: (
            r["metrics"]["hit_at_n_after"],
            r["metrics"]["token_f1"],
            r["metrics"]["grounded_bigram_recall"],
        ),
    )[:EVAL_MAX_FAILURE_EXAMPLES]

    return {
        "num_samples": count,
        "aggregates": aggregates,
        "reranker_lift": reranker_lift,
        "avg_total_latency_s": avg_total_latency,
        "failure_examples": [
            {
                "sample_id": f["sample_id"],
                "question": f["question"],
                "source_page": f["source_page"],
                "retrieved_pages_after": f["retrieved_pages_after"],
                "token_f1": f["metrics"]["token_f1"],
                "grounded_bigram_recall": f["metrics"]["grounded_bigram_recall"],
            }
            for f in failures
        ],
    }


def run_evaluation(pdf_path: str, eval_size: int = EVAL_DATASET_SIZE, top_k: int = SEARCH_TOP_K) -> dict:
    """Offline evaluation pipeline using synthetic Q/A data."""
    timings = {}

    with Timer("Eval: Parse + Embed (cached)") as t:
        processed_chunks = load_or_process(pdf_path)
    timings["load_or_process_s"] = t.elapsed

    with Timer("Eval: Weaviate Upload") as t:
        upload_to_weaviate(processed_chunks)
    timings["upload_s"] = t.elapsed

    with Timer("Eval: Synthetic Dataset") as t:
        current_hash = file_hash(pdf_path)
        samples = build_synthetic_dataset(processed_chunks, current_hash, dataset_size=eval_size)
    timings["synthetic_dataset_s"] = t.elapsed

    per_sample_results = []
    for i, sample in enumerate(samples, start=1):
        print(f"\n=== Evaluating sample {i}/{len(samples)} ===")
        per_sample_results.append(evaluate_one_sample(sample, top_k=top_k))

    metrics_summary = summarize_eval_results(per_sample_results)
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_path": pdf_path,
        "eval_size": len(samples),
        "top_k": top_k,
        "pipeline_timings": timings,
        "metrics_summary": metrics_summary,
        "per_sample_results": per_sample_results,
    }
    save_json(EVAL_RESULTS_PATH, payload)
    print(f"\nSaved evaluation results to {EVAL_RESULTS_PATH}")
    return payload


def print_eval_summary(payload: dict):
    summary = payload.get("metrics_summary", {})
    if not summary:
        print("\nNo evaluation summary available.")
        return
    agg = summary["aggregates"]
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Samples: {summary['num_samples']}")
    print(f"Hit@K (before rerank): {agg['hit_at_k_before']:.3f}")
    print(f"Hit@N (after rerank):  {agg['hit_at_n_after']:.3f}")
    print(f"MRR:                  {agg['reciprocal_rank']:.3f}")
    print(f"Token F1:             {agg['token_f1']:.3f}")
    print(f"Citation compliance:  {agg['citation_ok']:.3f}")
    print(f"Groundedness proxy:   {agg['grounded_bigram_recall']:.3f}")
    print(f"Reranker lift:        {summary['reranker_lift']:.3f}")
    print(f"Avg total latency:    {summary['avg_total_latency_s']:.2f}s")
    print("-" * 60)
    print("Top failure examples:")
    for failure in summary["failure_examples"]:
        print(
            f"  - {failure['sample_id']} | page={failure['source_page']} | "
            f"token_f1={failure['token_f1']:.3f} | grounded={failure['grounded_bigram_recall']:.3f}"
        )
    print("=" * 60)



# MAIN PIPELINE
def main():
    global SEARCH_TOP_K
    mode = (input("Choose mode [run/eval] (default=run): ") or "run").strip().lower()
    SEARCH_TOP_K = int(input("How many chunks to retrieve? [default=30]: ") or 30)

    if mode == "eval":
        eval_size = int(input(f"How many synthetic eval samples? [default={EVAL_DATASET_SIZE}]: ") or EVAL_DATASET_SIZE)
        eval_payload = run_evaluation(FILE_PATH, eval_size=eval_size, top_k=SEARCH_TOP_K)
        print_eval_summary(eval_payload)
        return

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