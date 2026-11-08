import json
from pathlib import Path
from evaluators.correctness import evaluate_correctness
from src.retriever import ask_query
from evaluators.faithfulness import evaluate_faithfulness
from evaluators.retrieval_relevance import evaluate_retrieval
from evaluators.citation_accuracy import evaluate_citation_accuracy
# Resolve paths relative to this file, not the cwd
EVALS_DIR = Path(__file__).parent


def load_goldens(path: str):
    with open(EVALS_DIR / path) as f:
        return json.load(f)


def generate_report(results: list[dict]) -> float:
    final_score = sum(r["score"] for r in results) / len(results)
    return final_score


def run_evaluation() -> float:
    dataset = load_goldens("dataset/golden_dataset_10.json")

    results = []

    for test_case in dataset:

        # Execute the RAG pipeline
        response = ask_query(test_case["question"])

        # Evaluate correctness using Groq as LLM judge
        # correctness = evaluate_correctness(
        #     question=test_case["question"],
        #     reference_answer=test_case["reference_answer"],
        #     generated_answer=response["content"],
        # )

        # res = {
        #     "id": test_case["id"],
        #     "score":  correctness["score"],
        #     "passed": correctness["passed"],
        #     "reason": correctness["reason"],
        # }

        # Evaluate faithfulness using Groq as LLM judge
        # faithfulness = evaluate_faithfulness(
        #     question=test_case["question"],
        #     retrieval_context=[doc.page_content for doc in response["relevant_chunks"]],
        #     generated_answer=response["content"],
        # )
        # res = {
        #     "id": test_case["id"],
        #     "score":  faithfulness["score"],
        #     "passed": faithfulness["passed"],
        #     "reason": faithfulness["reason"],
        # }
        # results.append(res)

        # retrieval = evaluate_retrieval(
        #     question=test_case["question"],
        #     retrieval_context=[
        #         doc.page_content for doc in response["relevant_chunks"]],
        #     expected_answer=test_case["reference_answer"],
        # )

        # results.append(retrieval)
        # print(retrieval)

        citation_accuracy = evaluate_citation_accuracy(
            retriever_response=response
        )

        results.append(citation_accuracy)
        print("ID:", test_case["id"])
        print(citation_accuracy)
        print("--------------------------------------")

    return results


print("final=", run_evaluation())
