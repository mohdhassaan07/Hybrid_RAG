# online/triad_worker.py
import asyncio
from dotenv import load_dotenv
from langsmith import AsyncClient
try:
    from .correctness import GroqJudge
except ImportError:
    from correctness import GroqJudge
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

load_dotenv()

PROJECT = "hybrid-rag-local"
JUDGE_MODEL = GroqJudge(model="openai/gpt-oss-120b")
THRESHOLD = 0.7
SAMPLE_RATE = 1          # ~30% of traffic — 3 metrics × several calls each = expensive
POLL_SECONDS = 60

# runs.query() returns an AsyncPaginator — use AsyncClient throughout
client = AsyncClient()


def _sampled(run):
    """Deterministic per-run sampling: same run always decided the same way."""
    return hash(str(run.id)) % 100 < SAMPLE_RATE * 100


async def _existing_keys(run):
    """Which feedback keys this run already has — so we never re-judge a key."""
    fb = client.list_feedback(run_ids=[run.id])
    return {f.key async for f in fb}


async def score_recent_traces():
    """One pass: read recent RagPipeline traces, score any missing triad metric, push feedback."""
    runs = client.runs.query(
        project_ids=["01a1d97b-7e3c-72a2-b5c9-959c8283f623"],
        filter='and(eq(run_type, "chain"), eq(status, "success"))',
    )

    async for run_summary in runs:
        # root-run guard: skip child spans (execution_order not supported in v2 filter)
        if run_summary.parent_run_ids is not None:
            continue

        if not _sampled(run_summary):
            continue

        # runs.query() returns lightweight summaries — outputs are empty.
        # Fetch the full run object to get inputs/outputs.
        run = await client.runs.retrieve(str(run_summary.id))
        outputs = run.outputs or {}

        # ask_query() returns "content" (the answer) and "relevant_chunks" (context docs)
        raw_answer = outputs.get("content")
        raw_chunks = outputs.get("relevant_chunks", [])
        query = (run.inputs or {}).get("query", "")

        # "content" can be a list of citation dicts or a plain string
        if isinstance(raw_answer, list):
            answer = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in raw_answer
            )
        else:
            answer = raw_answer

        # DeepEval expects retrieval_context as list[str]
        if raw_chunks and isinstance(raw_chunks[0], dict):
            # serialised Document dicts
            context = [c.get("page_content", "") for c in raw_chunks]
        elif raw_chunks and hasattr(raw_chunks[0], "page_content"):
            # LangChain Document objects
            context = [c.page_content for c in raw_chunks]
        else:
            context = [str(c) for c in raw_chunks]

        if not answer or not context:
            print(
                f"[skip] run {run.id} — missing answer or context (outputs keys: {list(outputs.keys())})")
            continue                          # nothing to judge

        already = await _existing_keys(run)

        # each metric: (feedback key, metric object, which inputs it needs)
        # built fresh per run so scores/reasons don't leak between traces
        jobs = [
            ("faithfulness",
             FaithfulnessMetric(threshold=THRESHOLD,
                                model=JUDGE_MODEL, include_reason=True),
             dict(input=query, actual_output=answer, retrieval_context=context)),

            ("answer_relevancy",
             AnswerRelevancyMetric(threshold=THRESHOLD,
                                   model=JUDGE_MODEL, include_reason=True),
             dict(input=query, actual_output=answer)),

            ("contextual_relevancy",
             ContextualRelevancyMetric(
                 threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
             dict(input=query, actual_output=answer, retrieval_context=context)),
        ]

        for key, metric, tc_kwargs in jobs:
            if key in already:
                continue                      # per-key dedup: skip only what's already scored

            try:
                metric.measure(LLMTestCase(**tc_kwargs))
                await client.create_feedback(
                    run_id=run.id,
                    key=key,
                    score=metric.score,
                    comment=metric.reason,
                )
            except Exception as e:
                # one metric failing shouldn't sink the other two for this trace
                print(f"[{key}] failed on run {run.id}: {e}")


if __name__ == "__main__":
    # live-in-a-terminal version. For production, delete the loop and let cron
    # call score_recent_traces() on a schedule instead.
    async def _loop():
        while True:
            print("evaluating...")
            await score_recent_traces()
            await asyncio.sleep(POLL_SECONDS)

    asyncio.run(_loop())
