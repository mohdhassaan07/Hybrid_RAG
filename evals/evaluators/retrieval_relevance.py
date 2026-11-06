from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase
try:
    from .correctness import GroqJudge
except ImportError:
    from correctness import GroqJudge  # when run directly as a script

def evaluate_retrieval(
    question: str,
    retrieval_context: list[str],
    expected_answer: str | None = None,
    groq_model: str = "openai/gpt-oss-120b",
    threshold: float = 0.7,
):
    if not retrieval_context:
        raise ValueError("No chunks were retrieved")

    judge = GroqJudge(model=groq_model)

    test_case = LLMTestCase(
        input=question,
        actual_output="Retrieval-only evaluation",
        expected_output=expected_answer,
        retrieval_context=retrieval_context,
    )

    # Measures how relevant the retrieved chunks are to the question
    relevancy = ContextualRelevancyMetric(
        threshold=threshold,
        model=judge,
        include_reason=True,
    )

    relevancy.measure(test_case)

    results = {
        "contextual_relevancy": {
            "score": relevancy.score,
            "passed": relevancy.is_successful(),
            "reason": relevancy.reason,
        }
    }

    # These two metrics need a reference answer
    if expected_answer:
        recall = ContextualRecallMetric(
            threshold=threshold,
            model=judge,
            include_reason=True,
        )

        precision = ContextualPrecisionMetric(
            threshold=threshold,
            model=judge,
            include_reason=True,
        )

        recall.measure(test_case)
        precision.measure(test_case)

        results["contextual_recall"] = {
            "score": recall.score,
            "passed": recall.is_successful(),
            "reason": recall.reason,
        }

        results["contextual_precision"] = {
            "score": precision.score,
            "passed": precision.is_successful(),
            "reason": precision.reason,
        }

    return results

if __name__ == "__main__":
    question = (
    "How does Project 4's human calibration influence the release process?"
    )

    expected_answer = (
    "Human labels establish a gold standard and allow judge-to-human agreement to be measured using weighted Cohen's kappa and Spearman correlation. The calibrated judge is then run through DeepEval in CI on each pull request, with explicit score thresholds causing builds to fail on regressions; the safety criterion cannot drop."
    )

    retrieval_context=["pull request.\n▸ Set explicit thresholds: no criterion may drop more than a defined amount, and the safety\ncriterion may not drop at all. Fail the build on regression.\n▸ Store every run in Postgres and chart scores over time in Streamlit, so quality has a trend\nline rather than an anecdote.\n▸ Demo it by opening a pull request that intentionally weakens a prompt and letting CI reject it\non camera.\nDone when\nA pull request that degrades output quality fails CI automatically, and the README reports judge\nto human agreement plus three measured bias numbers.\nResume line: Built an LLM as judge evaluation pipeline calibrated against N human labels at\nkappa X; quantified position, length, and self preference bias and gated releases on it in CI.\nWhere this usually goes wrong\n▸ No human baseline at all, which makes every judge score unfalsifiable.\n▸ One to ten scales, which produce inconsistent scores across runs and labelers.\n▸ Judging outputs with the same model that produced them",

    "PHASE 4\nTest the judge for its known biases\n▸ Position bias: run pairwise comparisons in both orders and measure how often the verdict\nflips. A flip rate above roughly ten percent means your pairwise setup is not trustworthy yet.\n▸ Length bias: hold quality constant and correlate score against output length. If longer wins,\nyour rubric is rewarding volume.\n▸ Self preference: have one model family judge outputs from itself and a competitor, then swap\nthe judge and compare verdicts on the same pairs.\n▸ Publish all three numbers. Most working engineers have never quantified any of them, which is\nprecisely why it stands out in an interview.\nPHASE 5\nGate releases on calibrated scores\n▸ Wire the judge into DeepEval and run it as a CI job against a fixed evaluation set on every\npull request.\n▸ Set explicit thresholds: no criterion may drop more than a defined amount, and the safety\ncriterion may not drop at all. Fail the build on regression.\n▸ Store every run in Postgres and chart scores over time in Streamlit, so quality has a trend\nline rather than an anecdote.\n▸ Demo it by opening a pull request that intentionally weakens a prompt and letting CI reject it\non camera.\nDone when\nA pull request that degrades output quality fails CI automatically, and the README reports judge\nto human agreement plus three measured bias numbers.\nResume line: Built an LLM as judge evaluation pipeline calibrated against N human labels at\nkappa X; quantified position, length, and self preference bias and gated releases on it in CI.\nWhere this usually goes wrong\n▸ No human baseline at all, which makes every judge score unfalsifiable.\n▸ One to ten scales, which produce inconsistent scores across runs and labelers.\n▸ Judging outputs with the same model that produced them.\n▸ Tuning the rubric on the same examples you report agreement on, which is leakage.\nStretch goals: Active learning to select the next most informative examples to label, score slicing\nby user cohort, and drift detection on sampled production traffic.\nbaswe.ai · 13",
    "baswe.Ai Engineer Accelerator™\nBuild These Six\nProjects\nA step by step build guide for the six portfolio projects\nthat prove you can do AI engineering work\nRAG Knowledge Graph RAG for Enterprise Data\nAGENTS Multi-Agent Research Assistant\nLLMOPS Self-Healing LLM Gateway\nEVALS LLM-as-Judge with Human Calibration\nFINE-TUNING Model Distillation Pipeline\nSAFETY Automated Red-Team Harness\nOne skill · one project · one proof\nbaswe.ai\nbaswe.ai · 1"
    ]

    results = evaluate_retrieval(
    question=question,
    retrieval_context=retrieval_context,
    expected_answer=expected_answer,
    )

    print(results)