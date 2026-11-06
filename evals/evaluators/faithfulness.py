from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
try:
    from .correctness import GroqJudge
except ImportError:
    from correctness import GroqJudge  # when run directly as a script
def evaluate_faithfulness(
    question:str,
    retrieval_context : list[str],
    generated_answer: str,
    groq_model: str = "openai/gpt-oss-120b",
    threshold: float = 0.7
    ) :

    if not retrieval_context:
        raise ValueError("Retrieved context cannot be empty")

    if not generated_answer.strip():
        raise ValueError("Generated answer cannot be empty")

    judge = GroqJudge(model=groq_model)

    metric = FaithfulnessMetric(
        threshold=threshold,
        model=judge,
        include_reason=True
    )
    test_case = LLMTestCase(
        input=question,
        actual_output=generated_answer,
        retrieval_context=retrieval_context
    )
    # To run metric as a standalone
    metric.measure(test_case)
    print(metric.score, metric.reason)
    # evaluate(test_cases=[test_case], metrics=[metric])

    if metric.score is None:
        raise RuntimeError("Faithfulness evaluation returned no score")
    
    return {
        "score": metric.score,
        "passed": metric.is_successful(),
        "reason": metric.reason,
    }

if __name__ == "__main__":
    res = evaluate_faithfulness(
        question="What information should be saved in Redis after every node in Project 2?",
        retrieval_context=["▸ Fetch and extract the page rather than trusting search snippets, which are truncated and often\nmisleading.\n▸ Deduplicate by URL and cap results per domain so a single site cannot dominate a report.\n▸ Handle each failure path explicitly: no results, rate limited, paywalled, timed out. Each\nreturns a structured status the supervisor can act on rather than an exception that kills the\nrun.\nPHASE 4\nPersist run state in Redis\n▸ Write graph state after every node: run id, plan, completed sub questions, findings so far,\nand cumulative token spend.\n▸ Now a crashed or redeployed process resumes instead of replaying, which is the difference\nbetween a demo and a service.\n▸ The same store gives you idempotency, since re entering a completed node returns the cached\nresult, and it powers a status endpoint so callers can see progress.\nPHASE 5\nShip a report endpoint with citations and a trace\n▸ POST /research returns a run id immediately. GET /research/{id} returns status and then the",
        "Packaging the work\nA finished project that nobody can evaluate in two minutes is an unfinished project. Every\nrepository gets the same five things, in this order.\n▸ One line at the top saying what it does and what it achieved, with the number in it.\n▸ The result immediately after: a benchmark table, a dashboard screenshot, or an embedded demo\nvideo.\n▸ An architecture diagram, one image, showing the data path rather than every class.\n▸ A short design decisions section covering the two or three real trade offs you made and what\nyou gave up.\n▸ A what did not work section. This is the part experienced engineers read first and the part\nnobody writes.\nTalking about them in the interview\nLead with the problem and the number, not the stack. Not I used LangGraph and Redis, but the runs\nwere losing four minutes of work on every deploy, so I made the state durable and added a cost\nceiling after one run cost eleven dollars. The second version invites the follow up questions you"
        ],
        generated_answer="After each node you should persist the graph state, specifically the run ID, the overall plan, the list of completed sub‑questions, the findings gathered so far, and the cumulative token spend (1)."
    )
    print(res)