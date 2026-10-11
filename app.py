"""
AI Resume Screening Assistant
LangChain + RAG (PDF loader -> splitter -> embeddings -> FAISS -> retriever
-> prompt -> LLM -> Pydantic output parser) wrapped in a Streamlit UI.
"""
import os
import tempfile
from typing import List, Literal

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------
# 1. Structured output schemas (used by the Output Parser)
# --------------------------------------------------------------------------


class CandidateEvaluation(BaseModel):
    candidate_name: str = Field(description="Candidate name or resume file name")
    match_score: int = Field(ge=0, le=100, description="Overall JD match score 0-100")
    matching_skills: List[str] = Field(description="JD skills found in the resume")
    missing_skills: List[str] = Field(description="JD skills NOT found in the resume")
    summary: str = Field(description="3-4 sentence candidate summary")
    strengths: List[str] = Field(description="Key strengths relative to the JD")
    weaknesses: List[str] = Field(description="Key weaknesses or gaps relative to the JD")
    recommendation: Literal["Strong Hire", "Hire", "Hold", "Reject"]
    justification: str = Field(description="Reasoning behind the recommendation")


class RankedCandidate(BaseModel):
    candidate_name: str
    match_score: int
    one_line_reason: str


class ComparisonResult(BaseModel):
    best_candidate: str = Field(description="Name of the best-fit candidate")
    ranking: List[RankedCandidate] = Field(description="All candidates, best first")
    justification: str = Field(description="Why the best candidate wins and key trade-offs")


eval_parser = PydanticOutputParser(pydantic_object=CandidateEvaluation)
compare_parser = PydanticOutputParser(pydantic_object=ComparisonResult)

# --------------------------------------------------------------------------
# 2. Prompt templates
# --------------------------------------------------------------------------

EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert technical recruiter. Evaluate ONE candidate against a Job "
            "Description using ONLY the resume excerpts provided in the context. "
            "Do not use outside knowledge and do not invent skills, employers, or "
            "experience. If something is not in the excerpts, treat it as missing. "
            "Score honestly: 90+ only for near-perfect fits.\n\n{format_instructions}",
        ),
        (
            "human",
            "CANDIDATE: {candidate}\n\n"
            "JOB DESCRIPTION:\n{jd}\n\n"
            "RESUME EXCERPTS (retrieved):\n{context}\n\n"
            "Return the evaluation.",
        ),
    ]
).partial(format_instructions=eval_parser.get_format_instructions())

COMPARE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a hiring manager. Using ONLY the candidate evaluations provided "
            "(which were derived from their resumes), rank the candidates for the job "
            "and pick the best one.\n\n{format_instructions}",
        ),
        (
            "human",
            "JOB DESCRIPTION:\n{jd}\n\nCANDIDATE EVALUATIONS (JSON):\n{evaluations}",
        ),
    ]
).partial(format_instructions=compare_parser.get_format_instructions())

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer recruiter questions using ONLY the resume excerpts in the context. "
            "Mention the candidate name for each fact. If the answer is not in the "
            "context, reply exactly: 'Not found in the uploaded resumes.'",
        ),
        ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}"),
    ]
)

# --------------------------------------------------------------------------
# 3. RAG helpers
# --------------------------------------------------------------------------

RETRIEVAL_FACETS = [
    "work experience, roles, employers and achievements",
    "technical skills, tools, programming languages and frameworks",
    "projects and publications",
    "education and certifications",
]


def load_resume(uploaded_file):
    """PDF Document Loader: returns a list of LangChain Documents tagged with candidate name."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    try:
        docs = PyPDFLoader(path).load()
    finally:
        os.remove(path)
    name = os.path.splitext(uploaded_file.name)[0]
    for d in docs:
        d.metadata["candidate"] = name
    return name, docs


def split_docs(docs):
    """Text Splitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.split_documents(docs)


def retrieve_context(vectorstore, jd, k_each=3):
    """Retriever: JD-driven query + facet queries, de-duplicated."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_each})
    queries = [jd[:1500]] + RETRIEVAL_FACETS
    seen, out = set(), []
    for q in queries:
        for d in retriever.invoke(q):
            if d.page_content not in seen:
                seen.add(d.page_content)
                out.append(d.page_content)
    return "\n\n---\n\n".join(out)


def make_llm(api_key, model):
    return ChatOpenAI(model=model, temperature=0, api_key=api_key)


# --------------------------------------------------------------------------
# 4. Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="AI Resume Screening Assistant", page_icon="📄", layout="wide")
st.title("📄 AI Resume Screening Assistant")
st.caption("LangChain + RAG | answers come only from the uploaded resumes")

with st.sidebar:
    st.header("Settings")
    default_key = os.getenv("OPENAI_API_KEY", "")
    try:
        default_key = st.secrets.get("OPENAI_API_KEY", default_key)
    except Exception:
        pass
    api_key = st.text_input("OpenAI API key", value=default_key, type="password")
    model_name = st.selectbox("LLM", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.markdown("---")
    st.markdown(
        "**Pipeline**  \nPDF loader → Text splitter → Embeddings → FAISS → "
        "Retriever → Prompt → LLM → Pydantic parser"
    )

col_a, col_b = st.columns(2)
with col_a:
    files = st.file_uploader("Upload resumes (PDF)", type=["pdf"], accept_multiple_files=True)
with col_b:
    jd = st.text_area("Job Description", height=220, placeholder="Paste the Job Description here...")

run = st.button("🔍 Evaluate Candidates", type="primary")

if run:
    if not api_key:
        st.error("Please provide an OpenAI API key in the sidebar.")
        st.stop()
    if not files or not jd.strip():
        st.error("Upload at least one resume PDF and enter a Job Description.")
        st.stop()

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        llm = make_llm(api_key, model_name)
        eval_chain = EVAL_PROMPT | llm | eval_parser

        evaluations, all_chunks = [], []
        progress = st.progress(0.0, text="Starting...")
        for i, f in enumerate(files):
            progress.progress(i / len(files), text=f"Analysing {f.name} ...")
            name, docs = load_resume(f)
            if not "".join(d.page_content for d in docs).strip():
                st.warning(f"{f.name}: no extractable text (scanned PDF?). Skipped.")
                continue
            chunks = split_docs(docs)
            all_chunks.extend(chunks)
            vs = FAISS.from_documents(chunks, embeddings)  # one index per candidate
            context = retrieve_context(vs, jd)
            result = eval_chain.invoke({"candidate": name, "jd": jd, "context": context})
            result.candidate_name = name
            evaluations.append(result)
        progress.progress(1.0, text="Done")

        comparison = None
        if len(evaluations) > 1:
            payload = "[" + ",".join(e.model_dump_json() for e in evaluations) + "]"
            comparison = (COMPARE_PROMPT | llm | compare_parser).invoke(
                {"jd": jd, "evaluations": payload}
            )

        combined_vs = FAISS.from_documents(all_chunks, embeddings) if all_chunks else None
        st.session_state["results"] = {
            "evaluations": evaluations,
            "comparison": comparison,
            "combined_vs": combined_vs,
            "jd": jd,
        }
    except Exception as e:  # surface API / parsing errors cleanly
        st.error(f"Something went wrong: {e}")
        st.stop()

data = st.session_state.get("results")
if data and data["evaluations"]:
    evaluations = data["evaluations"]
    tab1, tab2, tab3 = st.tabs(["📋 Evaluations", "🏆 Comparison & Best Candidate", "💬 Ask the Resumes"])

    with tab1:
        for e in sorted(evaluations, key=lambda x: x.match_score, reverse=True):
            with st.expander(f"{e.candidate_name} — {e.match_score}/100 — {e.recommendation}", expanded=True):
                c1, c2 = st.columns([1, 3])
                c1.metric("Match Score", f"{e.match_score}/100")
                c1.progress(e.match_score / 100)
                c1.markdown(f"**Recommendation:** {e.recommendation}")
                c2.markdown(f"**Summary:** {e.summary}")
                s1, s2 = st.columns(2)
                s1.markdown("**✅ Matching skills**")
                s1.write(", ".join(e.matching_skills) or "—")
                s2.markdown("**❌ Missing skills**")
                s2.write(", ".join(e.missing_skills) or "—")
                w1, w2 = st.columns(2)
                w1.markdown("**💪 Strengths**")
                for s in e.strengths:
                    w1.markdown(f"- {s}")
                w2.markdown("**⚠️ Weaknesses**")
                for w in e.weaknesses:
                    w2.markdown(f"- {w}")
                st.markdown(f"**Justification:** {e.justification}")

    with tab2:
        df = pd.DataFrame(
            [
                {
                    "Candidate": e.candidate_name,
                    "Score": e.match_score,
                    "Recommendation": e.recommendation,
                    "Missing skills": len(e.missing_skills),
                }
                for e in evaluations
            ]
        ).sort_values("Score", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Candidate")["Score"])
        cmp_ = data["comparison"]
        if cmp_:
            st.success(f"🏆 Best candidate: **{cmp_.best_candidate}**")
            for i, r in enumerate(cmp_.ranking, 1):
                st.markdown(f"{i}. **{r.candidate_name}** ({r.match_score}) — {r.one_line_reason}")
            st.markdown(f"**Justification:** {cmp_.justification}")
        else:
            st.info("Upload 2+ resumes to see a comparison.")

    with tab3:
        q = st.text_input("Ask a question (e.g. 'Who has Azure experience?')")
        if q and data["combined_vs"] is not None:
            retriever = data["combined_vs"].as_retriever(search_kwargs={"k": 8})
            docs = retriever.invoke(q)
            context = "\n\n".join(f"[{d.metadata['candidate']}] {d.page_content}" for d in docs)
            llm = make_llm(api_key, model_name)
            st.markdown((QA_PROMPT | llm | StrOutputParser()).invoke({"context": context, "question": q}))
            with st.expander("Retrieved sources"):
                for d in docs:
                    st.caption(f"{d.metadata['candidate']} (page {d.metadata.get('page', 0) + 1})")
                    st.write(d.page_content)
