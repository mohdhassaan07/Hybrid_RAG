
# online/triad_worker.py

import asyncio
import os
import logging
from dotenv import load_dotenv

from langsmith import AsyncClient

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

try:
    from .correctness import GroqJudge
except ImportError:
    from correctness import GroqJudge


load_dotenv()

# ---------------- CONFIGURATION ----------------

PROJECT_ID = "01a1d97b-7e3c-72a2-b5c9-959c8283f623"

THRESHOLD = 0.7
SAMPLE_RATE = 1.0       # 1.0 = 100%, 0.3 = 30%
POLL_SECONDS = 60

JUDGE_MODEL = GroqJudge(model="openai/gpt-oss-120b")

client = AsyncClient()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------- SAMPLING ----------------

def sampled(run_id):
    """Deterministic sampling for each trace."""
    import hashlib

    value = int(
        hashlib.md5(str(run_id).encode()).hexdigest(),
        16
    )

    return value % 10000 < SAMPLE_RATE * 10000


# ---------------- EXISTING FEEDBACK ----------------

async def existing_feedback(run_id):
    """Get existing metric keys to prevent duplicate evaluations."""

    feedback = client.list_feedback(
        run_ids=[str(run_id)]
    )

    return {
        item.key
        async for item in feedback
    }


# ---------------- OUTPUT EXTRACTION ----------------

def extract_answer(raw_answer):

    if isinstance(raw_answer, str):
        return raw_answer

    if isinstance(raw_answer, list):
        parts = []

        for item in raw_answer:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))

        return " ".join(parts)

    return str(raw_answer) if raw_answer else ""


def extract_context(raw_chunks):

    if not raw_chunks:
        return []

    context = []

    for chunk in raw_chunks:

        if isinstance(chunk, dict):
            content = (
                chunk.get("page_content")
                or chunk.get("text")
                or ""
            )

        elif hasattr(chunk, "page_content"):
            content = chunk.page_content

        else:
            content = str(chunk)

        if content:
            context.append(str(content))

    return context


# ---------------- METRIC EVALUATION ----------------

async def evaluate_metric(
    run_id,
    key,
    metric,
    test_case
):
    """Evaluate one metric and attach its score to LangSmith."""

    try:

        # DeepEval measure() is synchronous.
        # Run it in a worker thread to avoid blocking asyncio.
        await asyncio.to_thread(
            metric.measure,
            test_case
        )

        score = metric.score
        reason = metric.reason

        if score is None:
            raise ValueError("DeepEval returned an empty score")

        await client.create_feedback(
            run_id=str(run_id),
            session_id=PROJECT_ID,
            key=key,
            score=float(score),
            comment=str(reason or ""),
        )

        logger.info(
            "[SUCCESS] %s | Run: %s | Score: %.4f",
            key,
            run_id,
            score
        )

    except Exception:
        logger.exception(
            "[FAILED] %s | Run: %s",
            key,
            run_id
        )


# ---------------- PROCESS ONE TRACE ----------------

async def process_trace(run):

    run_id = run.id

    if not sampled(run_id):
        return

    inputs = run.inputs or {}
    outputs = run.outputs or {}

    query = (
        inputs.get("query")
        or inputs.get("question")
        or ""
    )

    raw_answer = outputs.get("content")
    raw_chunks = outputs.get("relevant_chunks", [])

    answer = extract_answer(raw_answer)
    context = extract_context(raw_chunks)

    if not query or not answer or not context:

        logger.warning(
            "[SKIPPED] Run %s | query=%s | answer=%s | context_chunks=%s | output_keys=%s",
            run_id,
            bool(query),
            bool(answer),
            len(context),
            list(outputs.keys())
        )

        return

    # Check which metrics have already been evaluated.
    already = await existing_feedback(run_id)

    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=context
    )

    # Create fresh metric instances for each run.
    jobs = {

        "faithfulness": FaithfulnessMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),

        "answer_relevancy": AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),

        "contextual_relevancy": ContextualRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
    }

    for key, metric in jobs.items():

        if key in already:
            logger.info(
                "[EXISTS] %s | Run: %s",
                key,
                run_id
            )
            continue

        await evaluate_metric(
            run_id=run_id,
            key=key,
            metric=metric,
            test_case=test_case
        )


# ---------------- FETCH RECENT TRACES ----------------

async def score_recent_traces():

    logger.info("Fetching recent LangSmith traces...")

    # IMPORTANT:
    # The v2 query API returns an async paginator.
    # Await the query before iterating.

    runs = await client.runs.query(
        project_ids=[PROJECT_ID],

        is_root=True,

        filter='eq(run_type, "CHAIN")',

        selects=[
            "ID",
            "RUN_TYPE",
            "STATUS",
            "INPUTS",
            "OUTPUTS",
            "TRACE_ID",
        ],

        page_size=100,
    )

    count = 0

    async for run in runs:

        count += 1

        try:
            await process_trace(run)

        except Exception:
            logger.exception(
                "Unexpected error processing run %s",
                run.id
            )

    logger.info(
        "Finished evaluation. Traces processed: %s",
        count
    )


# ---------------- WORKER LOOP ----------------

async def main():

    logger.info("DeepEval + LangSmith worker started.")

    try:

        while True:

            try:
                await score_recent_traces()

            except Exception:
                logger.exception("Worker iteration failed")

            await asyncio.sleep(POLL_SECONDS)

    finally:
        await client.aclose()


if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("Worker stopped.")
