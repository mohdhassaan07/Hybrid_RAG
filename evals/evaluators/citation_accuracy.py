import json
from groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class SupportVerdict(BaseModel):
    supported: bool
    reason: str


def ask_judge(client: Groq, model: str, system_prompt: str, payload: dict) -> dict:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


SUPPORT_PROMPT = """
Determine whether the cited source excerpts directly support the factual claim.

Use ONLY the supplied source excerpts.
Do not use outside knowledge.

Return JSON only:
{
  "supported": true | false,
  "reason": "one-sentence explanation"
}

Mark supported=true only when the supplied evidence clearly entails the claim.
""".strip()


def _build_citation_map(retriever_response: dict) -> dict[int, str]:
    """
    Build {citation_number (int) -> excerpt (str)} from the top-level
    'citations' list in the retriever response.

    Each entry looks like:
        {"number": 1, "excerpt": "...", "source": {...}}
    """
    return {
        cit["number"]: cit.get("excerpt", "")
        for cit in retriever_response.get("citations", [])
    }


def evaluate_citation_accuracy(
    retriever_response: dict,
    model: str = "openai/gpt-oss-120b",
) -> dict:
    """
    Evaluate citation accuracy using the structured retriever response.

    The response is expected to have:
      - content  : list[{"text": str, "citations": list[int]}]
            Each item is an answer segment (claim) with its citation numbers.
      - citations: list[{"number": int, "excerpt": str, "source": {...}}]
            Numbered source excerpts that back the claims.

    Returns
    -------
    dict:
        citation_accuracy  – fraction of cited claims supported by their sources
        citation_coverage  – fraction of all claims that carry ≥ 1 citation
        citation_validity  – fraction of citation refs that point to a known source
        details            – per-claim breakdown list
    """
    client = Groq()

    # ------------------------------------------------------------------ #
    # 1. Build the citation number → excerpt lookup                       #
    # ------------------------------------------------------------------ #
    citation_map: dict[int, str] = _build_citation_map(retriever_response)

    content_items: list[dict] = retriever_response.get("content", [])
    if not content_items:
        return {
            "citation_accuracy": 0.0,
            "citation_coverage": 0.0,
            "citation_validity": 0.0,
            "details": [],
            "error": "Empty content list — nothing to evaluate.",
        }

    results = []

    # ------------------------------------------------------------------ #
    # 2. Verify each claim against its cited source excerpts              #
    #    No LLM extraction needed — content already IS the claims list.  #
    # ------------------------------------------------------------------ #
    for item in content_items:
        claim_text: str = item.get("text", "").strip()
        cited_numbers: list[int] = item.get("citations", [])

        # Partition citation numbers into valid (exist in map) vs invalid
        valid_numbers = [n for n in cited_numbers if n in citation_map]
        invalid_numbers = [n for n in cited_numbers if n not in citation_map]

        supported = False
        reason = "No valid citations provided."

        if valid_numbers:
            cited_excerpts = {
                str(n): citation_map[n] for n in valid_numbers
            }

            verdict = SupportVerdict.model_validate(
                ask_judge(
                    client,
                    model,
                    SUPPORT_PROMPT,
                    {
                        "claim": claim_text,
                        "cited_sources": cited_excerpts,
                    },
                )
            )
            supported = verdict.supported
            reason = verdict.reason

        results.append(
            {
                "claim": claim_text,
                # Store as strings for consistency in downstream reporting
                "citation_ids":         [str(n) for n in cited_numbers],
                "invalid_citation_ids": [str(n) for n in invalid_numbers],
                # Fully supported = source backs it AND every ref is real
                "supported": supported and not invalid_numbers,
                "reason": reason,
            }
        )

    # ------------------------------------------------------------------ #
    # 3. Aggregate metrics                                                #
    # ------------------------------------------------------------------ #
    total_claims = len(results)

    cited_claims = [r for r in results if r["citation_ids"]]
    supported_claims = [r for r in cited_claims if r["supported"]]

    total_citation_refs = sum(len(r["citation_ids"]) for r in results)
    valid_citation_refs = sum(
        len(r["citation_ids"]) - len(r["invalid_citation_ids"])
        for r in results
    )

    return {
        # Of cited claims, what fraction are truly supported by their sources?
        "citation_accuracy": (
            len(supported_claims) / len(cited_claims) if cited_claims else 0.0
        ),
        # Of all claims, what fraction carry at least one citation?
        "citation_coverage": (
            len(cited_claims) / total_claims if total_claims else 1.0
        ),
        # Of all citation refs used, what fraction point to a real source?
        "citation_validity": (
            valid_citation_refs / total_citation_refs
            if total_citation_refs else 0.0
        ),
        "details": results,
    }


if __name__ == "__main__":

    res = evaluate_citation_accuracy(
        {
            "content": [
                {
                    "text": "Project 4 supplies a judge for open‑ended outputs that is used in Phase 4 benchmarking of the student model against the teacher.",
                    "citations": [
                        1
                    ]
                },
                {
                    "text": "Project 5 incorporates this judge from Project 4 when evaluating its fine‑tuned student model, allowing it to measure the quality of open‑ended student outputs in addition to standard task metrics.",
                    "citations": [
                        1,
                        3
                    ]
                }
            ],
            "confidence_score": 0.96,
            "citations": [
                {
                    "number": 1,
                    "excerpt": "▸ Watch for overfitting after two or three epochs on a small dataset, and for degradation if the\nlearning rate is too aggressive.\n▸ Train at least two configurations, for example LoRA rank eight against rank thirty two, so you\nhave a real trade off to discuss rather than a single lucky run.\nPHASE 4\nBenchmark student against teacher on three axes\n▸ Quality on the untouched test set using task metrics, plus the judge from Project 4 where the\noutput is open ended.\n▸ Cost per thousand requests: teac",
                    "source": {
                        "chunk_index": 34,
                        "file": "D:\\GenAI\\hybrid_RAG\\uploads\\BASWE_Build_These_Six_Projects.pdf",
                        "page": 15,
                        "section": ""
                    }
                },
                {
                    "number": 3,
                    "excerpt": "PROJECT 5 · FINE-TUNING\nModel Distillation Pipeline\nPython · Hugging Face · PyTorch · Claude API · W&B · vLLM\nEstimated build time: 4 to 5 weeks part time\nWhy this project signals\nThis is the project that lets you talk about unit economics, which is the fastest way to sound\nsenior. The output is not a model, it is a break even analysis: the request volume above which\nowning the model beats renting it. Very few candidates can produce that number from their own\nwork.\nPHASE 1\nPick one narrow task a",
                    "source": {
                        "chunk_index": 31,
                        "file": "D:\\GenAI\\hybrid_RAG\\uploads\\BASWE_Build_These_Six_Projects.pdf",
                        "page": 14,
                        "section": "Model Distillation Pipeline"
                    }
                }
            ],
        }
    )
    print(res)
