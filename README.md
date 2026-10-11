# AI Resume Screening Assistant (LangChain + RAG + Streamlit)

Recruiters upload resume PDFs and a Job Description. The app retrieves the relevant resume
passages (RAG) and produces a structured evaluation: match score (0-100), matching/missing
skills, summary, strengths, weaknesses and a hiring recommendation. Answers are grounded
only in the uploaded resumes.

## Architecture
PDF upload -> PyPDFLoader -> RecursiveCharacterTextSplitter (800/120)
-> OpenAI `text-embedding-3-small` -> FAISS (one index per candidate)
-> Retriever (JD query + 4 facet queries: experience, skills, projects, education)
-> ChatPromptTemplate -> ChatOpenAI (`gpt-4o-mini`, temperature 0)
-> PydanticOutputParser -> Streamlit UI

| Requirement | Implementation |
|---|---|
| LangChain | LCEL chains: `prompt \| llm \| parser` |
| LLM | OpenAI gpt-4o-mini |
| PDF Loader | `PyPDFLoader` |
| Text Splitter | `RecursiveCharacterTextSplitter` |
| Embeddings | `OpenAIEmbeddings` |
| Vector DB | FAISS |
| Retriever | `vectorstore.as_retriever()` |
| Prompt Template | `ChatPromptTemplate` (evaluation, comparison, Q&A) |
| Output Parser | `PydanticOutputParser` (CandidateEvaluation, ComparisonResult) |
| App | Streamlit (3 tabs: Evaluations, Comparison, Ask the Resumes) |

## Run locally (Windows CMD)
```
python -m venv resumeenv
resumeenv\Scripts\activate
pip install -r requirements.txt
set OPENAI_API_KEY=sk-...
streamlit run app.py
```
(You can also paste the key in the sidebar.)

## Deploy on Streamlit Community Cloud
1. Push `app.py` and `requirements.txt` to a public GitHub repo (never commit your key).
2. share.streamlit.io -> New app -> select repo, main file `app.py`.
3. Advanced settings -> Secrets: `OPENAI_API_KEY = "sk-..."`
4. Deploy and copy the public URL for submission.

## Limitations
Scanned (image-only) PDFs have no extractable text and are skipped; scores are LLM judgements
and should support, not replace, human review.
