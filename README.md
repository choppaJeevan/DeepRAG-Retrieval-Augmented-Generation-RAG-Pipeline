# RAG Optimized Pipeline

This project implements an optimized Retrieval-Augmented Generation (RAG) pipeline. It utilizes various advanced techniques such as Matryoshka Representation Learning (MRL), Semantic Chunking, Query Rewriting, and Cross-Encoder Re-ranking to deliver highly accurate, grounded answers.

## Architecture and Workflow

### Imports & Global Configuration
Sets up the environment and configures the core parameters for the RAG pipeline. It defines the target models (Nomic for embeddings, DeepSeek-R1 for generation, BGE for re-ranking) and establishes constraints like the target vector dimension (384 for MRL), context window sizes, and generation limits. It also loads environment variables, primarily the LlamaCloud API key.

### Utilities (`Timer`, `file_hash`, `get_cache_path`)
Provides helper functions to keep the pipeline efficient and track performance:
- **Timer**: A context manager used to measure and print the execution time of each major step in the pipeline.
- **file_hash**: Computes an MD5 hash of the source PDF. This acts as a fingerprint to detect if the file has changed since the last run.
- **get_cache_path**: Generates a standard file path in the `rag_cache` directory to store the processed JSON data.

### Step 1: Parsing & Chunking (`parse_and_chunk`)
Handles the ingestion of the PDF document. It uses **LlamaParse** to extract the text into Markdown format while preserving page numbers. It then applies a two-tier chunking strategy: a hard split to keep chunks manageable, followed by a **Semantic Chunker** that uses embeddings to find natural breaks in topics, preventing related sentences from being split apart.

### Batch Embedding & MRL (`batch_embed_and_slice`)
Converts the semantic chunks into vector representations. It processes all chunks in a single batch using Nomic embeddings to save time. Crucially, it applies **Matryoshka Representation Learning (MRL)** by slicing the high-dimensional vectors down to the target dimension (384) and normalizing them. This significantly reduces storage requirements and speeds up search without sacrificing accuracy.

### Caching Layer (`load_or_process`)
Acts as the central traffic controller for data ingestion. It checks the MD5 hash of the provided PDF against the local cache. 
- If the PDF hasn't changed, it instantly loads the pre-computed chunks and vectors from disk, bypassing the expensive parsing and embedding steps entirely. 
- If it's a new or modified file, it runs the parsing and embedding functions and caches the results.

### Step 2: Weaviate Upload (`upload_to_weaviate`)
Pushes the processed text chunks and their pre-computed MRL vectors into a local Weaviate vector database. It resets the specific collection if it already exists to ensure a fresh schema and uses a dynamic batch upload process for efficiency.

### Step 3: Query Rewriting (`rewrite_query`)
Optimizes the user's raw input. It uses DeepSeek-R1 to rewrite vague or simple questions into highly specific, search-optimized queries. This improves the vector database's ability to find relevant technical information. It also strips out the model's internal `<think>` reasoning tags to return just the clean query.

### Step 4: Query Embedding (`embed_query`)
Transforms the rewritten user query into a vector format. It applies the exact same MRL slicing and normalization logic used on the document chunks to ensure the query vector is mathematically compatible with the database vectors.

### Step 5: Vector Search (`weaviate_search`)
Performs the initial retrieval phase. It queries the Weaviate database using the embedded user query and retrieves the top K most similar chunks using Weaviate's fast HNSW index. It returns the raw text, source file name, and page numbers of the matches.

### Step 6: Cross-Encoder Re-ranking (`rerank_chunks`)
Refines the initial search results. It takes the top K chunks from Weaviate and pairs each one with the original user query. It passes these pairs through a BGE Cross-Encoder model, which is specifically trained to score the actual relevance between a query and a document. It sorts the chunks by this score and keeps only the absolute best (Top N) for the final answer.

### Step 7: Prompt Synthesis (`build_prompt`)
Constructs the final instructions for the generation model. It injects the top N re-ranked chunks into a strict system prompt that forces the LLM to act as a precise research assistant. The prompt demands inline citations, forbids outside knowledge, and requires the model to admit if the answer isn't in the text.

### Step 8: LLM Generation (`generate_answer`)
Produces the final output. It passes the synthesized prompt to the local DeepSeek-R1 model. After the model generates its response, this function cleans the output by removing the verbose `<think>` tags, returning only the grounded, final answer to the user.

### Main Pipeline (`main`)
The orchestrator of the entire script. It defines the execution sequence, prompts the user for the number of chunks to retrieve, captures the user's question, and ties all the previous steps together. Finally, it prints the model's answer alongside a detailed performance breakdown showing how long each step took.

## RAGAS Evaluation Pipeline

### Imports & Integrations
Imports the necessary evaluation tools from the RAGAS library alongside the core search and generation functions from your production RAG pipeline (`rag_optimized.py`). It also brings in `langchain_ollama` wrappers to connect the evaluation framework to your local models.

### Step 1: Initialization (INIT)
Sets up the local evaluation models. Instead of using paid external APIs, it configures an OpenAI client to point to your local Ollama instance (`http://localhost:11434/v1`). It establishes Mistral as the "LLM-as-a-Judge" to evaluate the responses, deliberately setting `num_predict=-1` to bypass output truncation limits, and initializes Nomic as the embedding model for relevance scoring.

### Step 2: Database Synchronization (SYNC)
Ensures the vector database is fully populated and up-to-date before evaluation begins. It calls the `load_or_process` and `upload_to_weaviate` functions from your main pipeline to guarantee the evaluator is testing against the correct document state.

### Step 3: Dataset Loading (LOAD)
Locates and loads the `eval_testset.json` file. This file contains the ground truth dataset, which includes the test questions and the expected reference answers needed to benchmark the RAG system's accuracy.

### Step 4: Pipeline Execution Loop (EXEC)
The core processing loop. It iterates through every question in the test set using a `tqdm` progress bar. For each question, it executes your entire production RAG flow:
- Embeds the query.
- Searches Weaviate.
- Re-ranks the top results.
- Synthesizes the prompt.
- Generates the final answer.

Once the answer is generated, it packages the original question, the retrieved context, the generated answer, and the ground-truth reference into a RAGAS `SingleTurnSample`.

### Step 5: Metric Binding & Evaluation (EVAL)
Configures the grading criteria. It explicitly binds your local Mistral and Nomic models to four specific RAGAS metrics to prevent the library from defaulting to OpenAI:
- **Faithfulness**: Checks if the answer is strictly derived from the retrieved context (no hallucinations).
- **Answer Relevancy**: Checks if the generated answer actually addresses the user's prompt.
- **Context Precision**: Evaluates if the most relevant chunks were ranked at the top of the search results.
- **Context Recall**: Measures if the retrieved context contains all the necessary information to answer the question based on the ground truth.

It then runs the `evaluate()` function asynchronously across the generated dataset to calculate the final scores.

### Step 6: Results Export
Aggregates the evaluation scores and converts the detailed, row-by-row results into a Pandas DataFrame. It exports this data to a new file named `eval_results.json` in the current directory, allowing you to easily review the system's performance and identify areas for improvement.
