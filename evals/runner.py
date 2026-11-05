import json
from pathlib import Path
from evaluators.correctness import evaluate_correctness
from src.retriever import ask_query


# Resolve paths relative to this file, not the cwd
EVALS_DIR = Path(__file__).parent


def load_goldens(path: str):
    with open(EVALS_DIR / path) as f:
        return json.load(f)


def generate_report(results: list[dict]) -> float:
    final_score = sum(r["correctness"]["score"] for r in results) / len(results)
    return final_score


def run_evaluation() -> float:
    dataset = load_goldens("dataset/golden_dataset_10.json")

    results = []

    for test_case in dataset:        

        # Execute the RAG pipeline
        response = ask_query(test_case["question"])
        
        # Evaluate correctness using Groq as LLM judge
        correctness = evaluate_correctness(
            question=test_case["question"],
            reference_answer=test_case["reference_answer"],
            generated_answer=response["content"],
        )

        # faithfulness = evaluate_faithfulness(...)
        # retrieval    = evaluate_retrieval(...)
        # citation     = evaluate_citations(...)
        res = {
            "id": test_case["id"],
            "correctness": {
                "score":  correctness["score"],
                "passed": correctness["passed"],
                "reason": correctness["reason"],
            },
        }

        results.append(res)
        print(res)

    return generate_report(results)


print("final=", run_evaluation())