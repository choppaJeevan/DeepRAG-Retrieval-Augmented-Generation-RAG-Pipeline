import os
import json
import time
from tqdm import tqdm
from openai import OpenAI

# Import from the production pipeline
from rag_optimized import (
    load_or_process,
    upload_to_weaviate,
    FILE_PATH,
    embed_query,
    weaviate_search,
    rerank_chunks,
    build_prompt,
    generate_answer
)

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import llm_factory
from langchain_ollama import OllamaEmbeddings

def main():
    print("=" * 60)
    print("INIT: Ollama Judge (Mistral) & Embeddings (Nomic)")
    print("=" * 60)
    
    # 1. Initialize OpenAI client pointing to local Ollama
    client = OpenAI(
        api_key="ollama", # API key is required by the SDK but ignored by Ollama
        base_url="http://localhost:11434/v1"
    )
    
    # 2. Setup RAGAS evaluators with local models
    # We use LangchainLLMWrapper with ChatOllama to force num_predict=-1, avoiding instructor truncation limits.
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model="mistral", temperature=0.0, num_predict=-1))
    evaluator_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    print("\n" + "=" * 60)
    print("SYNC: Validating Vector Database State")
    print("=" * 60)
    processed_chunks = load_or_process(FILE_PATH)
    upload_to_weaviate(processed_chunks)
    
    print("\n" + "=" * 60)
    print("LOAD: Evaluation Dataset")
    print("=" * 60)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testset_path = os.path.join(script_dir, "eval_testset.json")
    
    if not os.path.exists(testset_path):
        print(f"Error: {testset_path} not found! Please ensure it is created.")
        return
        
    with open(testset_path, "r") as f:
        testset = json.load(f)
    print(f"Loaded {len(testset)} test questions.")
        
    samples = []
    
    print("\n" + "=" * 60)
    print("EXEC: Running RAG Pipeline over Testset")
    print("=" * 60)
    
    for idx, item in enumerate(tqdm(testset)):
        query = item["question"]
        ground_truth = item.get("ground_truth", "")
        
        # Exec pipeline steps
        query_vector = embed_query(query)
        retrieved_chunks = weaviate_search(query_vector, top_k=30)
        
        # Suppress stdout strictly for reranking and generating in the loop unless we are debugging
        # because these outputs can clutter tqdm format
        top_chunks = rerank_chunks(query, retrieved_chunks, top_n=5)
        
        retrieved_contexts = [f"PAGE {chunk['page']}: {chunk['content']}" for chunk in top_chunks]
        
        prompt = build_prompt(query, top_chunks)
        answer = generate_answer(prompt)
        
        # Construct RAGAS Sample
        sample = SingleTurnSample(
            user_input=query,
            reference=ground_truth,
            retrieved_contexts=retrieved_contexts,
            response=answer
        )
        samples.append(sample)
        
    dataset = EvaluationDataset(samples=samples)
    
    print("\n" + "=" * 60)
    print("EVAL: Executing RAGAS Metrics")
    print("=" * 60)
    
    # Explicitly attach the llm and embeddings to the backward-compatible objects
    # to avoid the internal wrapper bug in ragas.evaluate() that strips out embed_query
    faithfulness.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings
    context_precision.llm = evaluator_llm
    context_recall.llm = evaluator_llm
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
    
    try:
        # Evaluate runs asynchronously internally, tqdm shows progress
        results = evaluate(
            dataset=dataset,
            metrics=metrics
        )
        
        print("\n" + "=" * 60)
        print("RESULTS: Aggregated Scores")
        print("=" * 60)
        print(results)
        
        # Save to JSON
        df = results.to_pandas()
        export_path = os.path.join(script_dir, "eval_results.json")
        df.to_json(export_path, orient="records", indent=4)
        print(f"\nDetailed results saved to {export_path}")
        
    except Exception as e:
        print(f"\nEvaluation Failure: {e}")

if __name__ == "__main__":
    main()
