import os
from openai import OpenAI, AsyncOpenAI
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams


GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = "openai/gpt-oss-120b"):
        self.model_name = model
        api_key = os.environ.get("GROQ_API_KEY")
        self._client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    def load_model(self):
        return self._client

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        response = await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return self.model_name


def evaluate_correctness(
    question:str,
    reference_answer: str,
    generated_answer: str,
    groq_model: str = "openai/gpt-oss-120b",
    threshold: float = 0.7,
) -> dict:
    """
    Evaluate a single generated answer against a golden reference using Groq as the LLM judge.

    Returns a dict with: score, passed, reason.
    """
    judge = GroqJudge(model=groq_model)

    metric = GEval(
        name="Correctness",
        model=judge,
        threshold=threshold,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=[
            "Read the question (input) and the reference_answer (expected output) carefully.",
            "Compare the generated_answer (actual output) to the reference_answer.",
            "Award a high score (close to 1) when the generated answer is factually "
            "consistent with the reference answer and covers all key points.",
            "Deduct points for factual errors, hallucinated details, or missing "
            "critical information present in the reference answer.",
            "Minor paraphrasing or additional true context should not reduce the score.",
        ],
        verbose_mode=False,
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=generated_answer,
        expected_output=reference_answer,
    )

    metric.measure(test_case)

    return {
        
        "score": metric.score,
        "passed": metric.is_successful(),
        "reason": metric.reason,
    }


if __name__ == "__main__":
    result = evaluate_correctness(
        question="How does Project 6 turn a security failure into a regression test, and how is continued improvement demonstrated?",
        reference_answer=
            "A confirmed failure is filed with a minimal reproduction, payload, response, and suggested mitigation. After the fix, that finding remains permanently in the suite as a regression case. The dashboard tracks pass rate by attack category and regression count, and the next scheduled run demonstrates that the vulnerability stays closed."
        ,
        generated_answer=
            "Project 6 converts any confirmed security failure into a lasting regression test by first generating a detailed finding – including a minimal reproduction, the exact payload, the response, and a mitigation – and opening a GitHub issue (1). After the fix is applied, the finding is never removed; instead it is added to the permanent regression suite so future runs automatically verify the vulnerability stays closed (1). Continued improvement is shown through scheduled runs that feed a dashboard tracking metrics such as findings by severity over time, mean time to close, pass rate by attack category, and a regression count that must remain zero. The demo’s closed‑loop – a vulnerability discovered, a fix shipped, and a subsequent run confirming the issue stays resolved – provides a clear trend line of improving pass rates and zero regressions (1).",
        
    )

    print(f"Score  : {result['score']:.2f}")
    print(f"Passed : {result['passed']}")
    print(f"Reason : {result['reason']}")
