"""
Hybrid RAG Query Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import re
import os
import sys

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid RAG Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Global */
body { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0f1117; }

/* Citation badge */
.citation-badge {
    display: inline-block;
    background: #4f8ef7;
    color: white;
    border-radius: 50%;
    width: 22px;
    height: 22px;
    text-align: center;
    line-height: 22px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    margin: 0 2px;
    vertical-align: middle;
}
.citation-badge:hover { background: #2563eb; }

/* Answer paragraph */
.answer-para {
    font-size: 16px;
    line-height: 1.8;
    margin-bottom: 10px;
    padding: 12px 16px;
    border-left: 3px solid #4f8ef7;
    background: #1a1f2e;
    border-radius: 0 8px 8px 0;
}

/* Chunk card */
.chunk-card {
    background: #1a1f2e;
    border: 1px solid #2d3348;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
    position: relative;
}
.chunk-rank {
    position: absolute;
    top: 10px;
    right: 14px;
    background: #252b3b;
    color: #a0aec0;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 20px;
}
.chunk-meta {
    font-size: 11px;
    color: #718096;
    margin-top: 6px;
}
.chunk-content {
    font-size: 13px;
    color: #cbd5e0;
    margin-top: 8px;
    white-space: pre-wrap;
}

/* Confidence bar */
.conf-label {
    font-size: 13px;
    color: #a0aec0;
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
}

/* Low-confidence banner */
.low-conf-banner {
    background: #2d1b1b;
    border: 1px solid #c53030;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 16px;
    color: #fc8181;
}

/* Citation popover area */
.citation-card {
    background: #1e2535;
    border: 1px solid #3a4259;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.citation-num {
    font-weight: 700;
    color: #4f8ef7;
    font-size: 15px;
}
.citation-source {
    font-size: 11px;
    color: #718096;
    margin-top: 4px;
}
.citation-excerpt {
    font-size: 12px;
    color: #a0aec0;
    margin-top: 8px;
    border-left: 2px solid #4f8ef7;
    padding-left: 10px;
    white-space: pre-wrap;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ─────────────────────────────────────────────────────────────────────
def inject_citation_badges(text: str) -> str:
    """Replace [1], [2]… with styled badge spans."""
    return re.sub(
        r"\[(\d+)\]",
        r'<span class="citation-badge" title="See citation \1">\1</span>',
        text,
    )


def render_confidence_bar(label: str, value: float, color: str = "#4f8ef7"):
    pct = int(value * 100)
    bar_html = f"""
<div class="conf-label"><span>{label}</span><span>{pct}%</span></div>
<div style="background:#252b3b;border-radius:6px;height:10px;margin-bottom:12px;">
  <div style="width:{pct}%;background:{color};height:10px;border-radius:6px;
       transition:width .4s ease;"></div>
</div>
"""
    st.markdown(bar_html, unsafe_allow_html=True)


def score_color(v: float) -> str:
    if v >= 0.75:
        return "#48bb78"   # green
    if v >= 0.5:
        return "#ecc94b"   # yellow
    return "#fc8181"       # red


def run_query(query: str):
    """Call ask_query and cache the result in session_state."""
    from src.retriever import ask_query  # local import avoids startup cost
    with st.spinner("Retrieving & generating answer…"):
        return ask_query(query)


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/search.png",
        width=56,
    )
    st.title("Hybrid RAG")
    st.caption("BM25 + Vector + RRF pipeline")
    st.divider()

    st.markdown("**Settings**")
    show_raw_chunks = st.toggle("Show all retrieved chunks", value=True)
    show_chunk_content = st.toggle("Show chunk text preview", value=True)
    max_preview_chars = st.slider(
        "Chunk preview length (chars)", 100, 800, 300, step=50
    )
    st.divider()
    st.markdown(
        "<span style='font-size:11px;color:#718096;'>"
        "FastAPI backend: `uvicorn main:app`<br>"
        "Vector DB: Qdrant on localhost:6333"
        "</span>",
        unsafe_allow_html=True,
    )

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## 🔍 Query Dashboard")
st.markdown(
    "Ask questions against your ingested documents. "
    "Answers include **clickable citations**, ranked chunks, "
    "and per-dimension confidence scores."
)
st.divider()

# ── Query input ──────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Your question",
        placeholder="e.g. What are the key retrieval strategies used?",
        label_visibility="collapsed",
    )
with col_btn:
    submit = st.button("Ask ✦", use_container_width=True, type="primary")

# Example queries
with st.expander("💡 Example questions", expanded=False):
    examples = [
        "Which two projects does the guide recommend building if there is time for only two?",
        "What is reciprocal rank fusion and how does it work?",
        "What chunking strategies are discussed in the documents?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            query = ex
            submit = True

st.divider()

# ── Main logic ───────────────────────────────────────────────────────────────
if submit and query.strip():
    result = run_query(query.strip())

    # ── Determine low-confidence path ────────────────────────────────────────
    confidence_score = result.get("confidence_score", 0.0)
    is_low_conf = confidence_score < 0.5
    content = result.get("content", [])
    citations = result.get("citations", [])
    chunks = result.get("relevant_chunks", [])

    # ── Layout: 3 columns ─────────────────────────────────────────────────────
    col_answer, col_conf = st.columns([3, 1], gap="large")

    # ════════════════════════════════════════════════════════════════
    # LEFT – Generated Answer
    # ════════════════════════════════════════════════════════════════
    with col_answer:
        st.markdown("### 💬 Generated Answer")

        if is_low_conf:
            # Low-confidence fallback response
            st.markdown(
                f'<div class="low-conf-banner">'
                f"⚠️ <strong>Low Confidence</strong> — The model could not answer fully "
                f"(confidence: {int(confidence_score*100)}%). "
                f"Showing what was found."
                f"</div>",
                unsafe_allow_html=True,
            )
            if isinstance(content, str):
                st.markdown(
                    f'<div class="answer-para">{inject_citation_badges(content)}</div>',
                    unsafe_allow_html=True,
                )
            relevant_fragments = result.get("relevant_fragments", [])
            missing_info = result.get("missing_information", [])
            if relevant_fragments:
                st.markdown("**Relevant fragments found:**")
                for frag in relevant_fragments:
                    st.markdown(f"- {frag}")
            if missing_info:
                st.markdown("**Missing information:**")
                for item in missing_info:
                    st.markdown(f"- _{item}_")
        else:
            # Normal structured answer with inline citations
            if isinstance(content, list):
                for claim in content:
                    text = claim.get("text", "")
                    claim_citations = claim.get("citations", [])
                    badges = " ".join(
                        f'<span class="citation-badge" title="See citation {c}">{c}</span>'
                        for c in claim_citations
                    )
                    st.markdown(
                        f'<div class="answer-para">'
                        f"{inject_citation_badges(text)} {badges}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            elif isinstance(content, str):
                st.markdown(
                    f'<div class="answer-para">{inject_citation_badges(content)}</div>',
                    unsafe_allow_html=True,
                )

        # ── Citations panel ────────────────────────────────────────────────
        if citations:
            st.markdown("---")
            st.markdown("#### 📎 Citations")
            for cit in citations:
                num = cit.get("number", "?")
                excerpt = cit.get("excerpt", "")
                source = cit.get("source", {})
                file_path = source.get("file", "")
                page = source.get("page")
                section = source.get("section", "")
                chunk_index = source.get("chunk_index")

                file_name = os.path.basename(
                    file_path) if file_path else "Unknown source"
                meta_parts = [f"📄 {file_name}"]
                if page is not None:
                    meta_parts.append(f"Page {page}")
                if section:
                    meta_parts.append(f"§ {section}")
                if chunk_index is not None:
                    meta_parts.append(f"Chunk #{chunk_index}")
                meta_str = " · ".join(meta_parts)

                # Clickable file link if path exists
                link_html = ""
                if file_path and os.path.exists(file_path):
                    link_html = (
                        f' <a href="file:///{file_path.replace(chr(92), "/")}" '
                        f'target="_blank" style="font-size:11px;color:#4f8ef7;">'
                        f"↗ Open file</a>"
                    )

                preview = excerpt[:400] + ("…" if len(excerpt) > 400 else "")

                st.markdown(
                    f'<div class="citation-card">'
                    f'<span class="citation-num">[{num}]</span>'
                    f'<div class="citation-source">{meta_str}{link_html}</div>'
                    f'<div class="citation-excerpt">{preview}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ════════════════════════════════════════════════════════════════
    # RIGHT – Confidence Scores
    # ════════════════════════════════════════════════════════════════
    with col_conf:
        st.markdown("### 📊 Confidence")

        # Overall gauge
        overall_color = score_color(confidence_score)
        overall_pct = int(confidence_score * 100)
        st.markdown(
            f"""
<div style="text-align:center;padding:20px 10px;background:#1a1f2e;
            border-radius:12px;border:1px solid #2d3348;margin-bottom:20px;">
  <div style="font-size:38px;font-weight:800;color:{overall_color};">{overall_pct}%</div>
  <div style="font-size:12px;color:#718096;margin-top:4px;">Overall Confidence</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("**Breakdown**")

        # --- Dimension scores ---
        # We derive approximate per-dimension scores from the overall score
        # and observable signals. These can be replaced by actual LLM-returned
        # dimension scores if the prompt is extended to emit them.
        num_citations = len(citations)
        num_claims = len(content) if isinstance(content, list) else 1

        # Citation Coverage: fraction of claims that have at least one citation
        if isinstance(content, list) and num_claims > 0:
            cited_claims = sum(
                1 for c in content if c.get("citations")
            )
            citation_coverage = round(cited_claims / num_claims, 2)
        else:
            citation_coverage = min(confidence_score + 0.05, 1.0)

        # Retrieval Confidence: driven by overall score with slight boost when chunks found
        retrieval_conf = min(
            round(confidence_score * 1.1 + (0.05 if chunks else 0), 2), 1.0
        )

        # Answer Completeness: ratio of unique cited chunks to total retrieved chunks
        total_chunks = len(chunks) if chunks else 1
        answer_completeness = min(
            round(num_citations / total_chunks +
                  confidence_score * 0.5, 2), 1.0
        )

        render_confidence_bar(
            "🔎 Retrieval Confidence",
            retrieval_conf,
            score_color(retrieval_conf),
        )
        render_confidence_bar(
            "📎 Citation Coverage",
            citation_coverage,
            score_color(citation_coverage),
        )
        render_confidence_bar(
            "✅ Answer Completeness",
            answer_completeness,
            score_color(answer_completeness),
        )

        st.divider()
        st.markdown("**Retrieved chunks**")
        st.metric("Total chunks", len(chunks))
        st.metric("Used in citations", num_citations)
        missing_info_list = result.get("missing_information", [])
        if missing_info_list:
            st.metric("Missing info items", len(missing_info_list))

    # ════════════════════════════════════════════════════════════════
    # BOTTOM – Retrieved Chunks ranked by relevance
    # ════════════════════════════════════════════════════════════════
    if show_raw_chunks and chunks:
        st.divider()
        st.markdown("### 🗂️ Retrieved Chunks (ranked by RRF score)")
        st.caption(
            "Chunks are ordered by Reciprocal Rank Fusion (BM25 + Vector). "
            "Rank 1 = most relevant."
        )

        for rank, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata if hasattr(chunk, "metadata") else {}
            file_path = meta.get("source", "")
            file_name = os.path.basename(file_path) if file_path else "Unknown"
            page = meta.get("page")
            section = meta.get("section", "")
            chunk_index = meta.get("chunk_index")
            # from cross-encoder if used
            relevance_score = meta.get("relevence_score")

            # Is this chunk actually cited?
            cited_numbers = {c.get("number") for c in citations}
            # find which citation number maps to this chunk
            is_cited = any(
                c.get("excerpt", "")[:80] == chunk.page_content[:80]
                for c in citations
            )

            cited_badge = (
                ' <span style="background:#276749;color:#9ae6b4;font-size:10px;'
                'padding:2px 8px;border-radius:20px;font-weight:600;">CITED</span>'
                if is_cited
                else ""
            )
            relevance_badge = (
                f' <span style="background:#2a3550;color:#90cdf4;font-size:10px;'
                f'padding:2px 8px;border-radius:20px;">LLM score: {relevance_score}</span>'
                if relevance_score
                else ""
            )

            meta_parts = [f"📄 {file_name}"]
            if page is not None:
                meta_parts.append(f"p.{page}")
            if section:
                meta_parts.append(f"§ {section}")
            if chunk_index is not None:
                meta_parts.append(f"chunk #{chunk_index}")
            meta_str = " · ".join(meta_parts)

            content_preview = (
                chunk.page_content[:max_preview_chars]
                + ("…" if len(chunk.page_content) > max_preview_chars else "")
                if show_chunk_content
                else ""
            )
            content_html = (
                f'<div class="chunk-content">{content_preview}</div>'
                if show_chunk_content
                else ""
            )

            st.markdown(
                f'<div class="chunk-card">'
                f'<div class="chunk-rank">Rank {rank}</div>'
                f"<strong style='font-size:14px;color:#e2e8f0;'>"
                f"Chunk {rank}"
                f"</strong>{cited_badge}{relevance_badge}"
                f'<div class="chunk-meta">{meta_str}</div>'
                f"{content_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

elif submit and not query.strip():
    st.warning("Please enter a question before submitting.")

else:
    # Empty state
    st.markdown(
        """
<div style="text-align:center;padding:60px 20px;color:#4a5568;">
  <div style="font-size:48px;">🔍</div>
  <div style="font-size:18px;margin-top:12px;">Ask a question to get started</div>
  <div style="font-size:13px;margin-top:6px;">
    Results will show the generated answer, ranked chunks, and confidence scores.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
