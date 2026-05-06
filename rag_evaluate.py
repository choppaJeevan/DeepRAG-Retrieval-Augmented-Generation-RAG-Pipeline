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
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model="mistral", temperature=0.0, num_predict=-1, timeout=600, format="json"))
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
    
    # Check if we already have generated answers cached!
    cache_path = os.path.join(script_dir, "eval_generations_cache.json")
    if os.path.exists(cache_path):
        print("\n" + "=" * 60)
        print("LOAD: Using cached generated answers")
        print("=" * 60)
        with open(cache_path, "r") as f:
            cache_data = json.load(f)
        for item in cache_data:
            samples.append(SingleTurnSample(
                user_input=item["user_input"],
                reference=item["reference"],
                retrieved_contexts=item["retrieved_contexts"],
                response=item["response"]
            ))
        print(f"Loaded {len(samples)} cached generations. Skipping generation step!")
    else:
        print("\n" + "=" * 60)
        print("EXEC: Running RAG Pipeline over Testset")
        print("=" * 60)
        
        for idx, item in enumerate(tqdm(testset)):
            query = item["question"]
            ground_truth = item.get("ground_truth", "")
            
            # Exec pipeline steps
            query_vector = embed_query(query)
            retrieved_chunks = weaviate_search(query_vector, top_k=30)
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
            
        # SAVE CACHE SO WE NEVER HAVE TO RERUN THIS AGAIN
        try:
            cache_data = [
                {
                    "user_input": s.user_input,
                    "reference": s.reference,
                    "retrieved_contexts": s.retrieved_contexts,
                    "response": s.response
                } for s in samples
            ]
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=4)
            print(f"\nSaved generation cache to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
            
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
        # Evaluate runs asynchronously internally. Ollama often times out on parallel requests.
        # RAGAS doesn't always expose max_workers cleanly via evaluate, but we can set raise_exceptions=False
        # to ensure it finishes even if some fail.
        from ragas.run_config import RunConfig
        
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=RunConfig(max_workers=1, timeout=7200, max_retries=1)
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
