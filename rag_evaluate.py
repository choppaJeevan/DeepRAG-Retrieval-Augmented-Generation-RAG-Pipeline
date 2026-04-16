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
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall
)
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings

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
    evaluator_llm = llm_factory("mistral", client=client)
    evaluator_embeddings = RagasOpenAIEmbeddings(model="nomic-embed-text", client=client)
    
    print("\n" + "=" * 60)
    print("SYNC: Validating Vector Database State")
    print("=" * 60)
    processed_chunks = load_or_process(FILE_PATH)
    upload_to_weaviate(processed_chunks)
    
    print("\n" + "=" * 60)
    print("LOAD: Evaluation Dataset")
    print("=" * 60)
    if not os.path.exists("eval_testset.json"):
        print("Error: eval_testset.json not found! Please ensure it is created.")
        return
        
    with open("eval_testset.json", "r") as f:
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
        
        retrieved_contexts = [chunk["content"] for chunk in top_chunks]
        
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
    
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecisionWithoutReference(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm)
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
        export_path = "eval_results.json"
        df.to_json(export_path, orient="records", indent=4)
        print(f"\nDetailed results saved to {export_path}")
        
    except Exception as e:
        print(f"\nEvaluation Failure: {e}")

if __name__ == "__main__":
    main()
