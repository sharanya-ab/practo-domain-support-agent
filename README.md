# Practo Domain Support Agent — Healthcare Track

**Track completed: Practo (Healthcare).** This repository implements the complete capstone workflow across dataset design, local RAG, CrewAI orchestration, AutoGen review, FastAPI deployment, evaluation, observability, caching, guardrails, and governance.

## Reproducible dataset-design choices (Part 1, Task 1)

- Random seed: `42`
- Records: `50`
- Category weights: General Medicine 30%, Cardiology 18%, Dermatology 18%, Pediatrics 17%, Orthopedics 17%.
- Status weights: Scheduled 35%, Completed 30%, Cancelled 12%, No-Show 8%, Rescheduled 15%.
- Consultation fee range: ₹400–₹1,800, selected to represent a plausible synthetic range across general and specialist consultations; these are fabricated demonstration values, not Practo prices.
- Follow-up probability: 20%.
- Fees are selected by specialty band; category and status coverage are validated after generation. No records are hand-edited.

## Quick start — Windows 11 / Python 3.11

1. Extract this ZIP.
2. Double-click `setup.bat`. This creates `.venv`, installs pinned dependencies, and downloads the free local embedding model once.
3. Double-click `run_all.bat`. It runs dataset validation, RAG indexing and threshold calibration, evaluation, guardrail/cache/governance demonstrations, CrewAI/AutoGen demos, and writes evidence to `artifacts/`.
4. Double-click `run_server.bat` to start FastAPI at `http://127.0.0.1:8000`. Open `/docs` to test endpoints.

Alternatively:
```bat
setup.bat
run_all.bat
run_server.bat
```

Activate manually with `.venv\Scripts\activate.bat` if desired.


## Corporate Windows environment

The project was developed and tested on Windows 11 using Python 3.11.3, VS Code, and Git Bash.

Some corporate networks use HTTPS inspection with certificates trusted by Windows but not automatically recognized by Python's default certificate bundle.

For this environment, install `truststore` in the virtual environment:

```bash
.venv/Scripts/python.exe -m pip install truststore
```

To run the complete capstone workflow:

```bash
.venv/Scripts/python.exe run_all.py
```

The runner initializes `truststore` before importing the RAG and agent components, allowing Python to use the operating system's trusted certificate authorities.

To start the FastAPI server in a corporate Windows environment where HTTPS inspection affects Python certificate validation, use:

```bash
.venv/Scripts/python.exe -c "import truststore; truststore.inject_into_ssl(); import uvicorn; uvicorn.run('api.main:app', host='127.0.0.1', port=8000)"
```

Open `http://127.0.0.1:8000/docs` to access the Swagger UI.

If port `8000` is already occupied, change `port=8000` to `port=8001` and open `http://127.0.0.1:8001/docs`.

This initializes the operating system's trusted certificate authorities before starting the API server, without disabling TLS certificate verification.

To run the tests independently:

```bash
.venv/Scripts/python.exe -c "import truststore, pytest; truststore.inject_into_ssl(); raise SystemExit(pytest.main(['tests/', '-q']))"
```

To run the end-to-end evaluation independently:

```bash
.venv/Scripts/python.exe -m evaluation.evaluate_agent
```

The end-to-end evaluation writes its results to `artifacts/agent_evaluation.json`.

Do not disable TLS certificate verification to resolve network errors. If the corporate certificate is not trusted by Windows, contact the organization's IT team.

## Runtime and network statement

The graded execution mode is `MOCK_LLM=true`. No model API key, paid account, or remote LLM call is used. Set `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` **before importing/running CrewAI** to prevent telemetry. The setup step needs internet once to install Python packages and fetch the public SentenceTransformers embedding model; after that, embeddings and vector search run locally. The agent inference/review path itself is deterministic and offline. For a fully air-gapped machine, pre-download the packages and model directory, then point `EMBEDDING_MODEL` to that local directory.

## Architecture

- `dataset.py`: seeded 50-row synthetic appointment generator and validations.
- `knowledge_base/`: 12 original policy documents, 2–5 sentences each.
- `rag/`: fixed-overlap and sentence chunkers; two independent ChromaDB collections; local SentenceTransformers embeddings; threshold calibration; grounded answer; document-level precision/recall.
- `agents/`: custom CrewAI `BaseLLM` extension, three-agent Crew, tools, Pydantic response schema, and LangChain session history.
- `review/`: AutoGen AgentChat RoundRobinGroupChat with two reviewer agents and a Pydantic verdict.
- `guardrails/`: fixed-format contact-number masking, prompt-injection detection, and retrieval-groundedness refusal.
- `governance/`: least-autonomy check, risk classification, request budget, normalized-query cache.
- `api/`: FastAPI `/ask`, `/add-document`, `/health`, and WebSocket `/ws/chat`; JSONL audit log.
- `evaluation/`: 15-query test set, four score dimensions, average calculations.
- `scripts/`: end-to-end orchestration and acceptance demonstrations.
- `artifacts/`: generated JSON/CSV/JSONL transcripts and reports.

## Important synthetic-data and safety note

All patient names, record IDs, contact numbers, diagnoses, and insurance identifiers in this repository are fabricated. The system masks fixed-format 10-digit contact numbers (including optional +91 prefix) before model processing and before logging. Free-text names, conditions, and insurance IDs are not reliably maskable without a named-entity system and are explicitly out of scope; do not enter real patient data. This is a policy-support demonstration, not a medical diagnosis or clinical decision system.

## Dataset and escalation formula

`dataset.py` deterministically generates 50 appointments. Escalation score:
`0.60 * int(follow_up_required) + 0.40 * (1 - days_since_created / 30)`.
The recommended escalation threshold is derived at runtime from the 80th percentile of the generated score distribution; the exact numeric value is written to `artifacts/dataset_report.json`.

## RAG threshold calibration

The threshold is not a hard-coded tutorial constant. `rag/build_index.py` measures top-1 cosine similarities for at least five in-scope and two out-of-scope probes. It selects the midpoint between the minimum in-scope score and maximum out-of-scope score only when the observed clusters separate; otherwise it stops with a diagnostic rather than silently choosing a misleading threshold. Exact measured values and the chosen threshold are written to `artifacts/rag_calibration.json`. Re-run the script after changing the knowledge base or embedding model.


## Evaluation interpretation

The project includes three complementary validation approaches:

1. **RAG retrieval evaluation:** A 15-query dataset covering 12 policy questions, one out-of-scope question, one unsafe medical request, and one privacy request. The reported accuracy, grounding, completeness, and safety values are simplified demonstration scores, not independently validated measures of response quality.

2. **End-to-end agent evaluation:** Six scenarios exercise the FastAPI workflow, including cancellation policy retrieval, appointment lookup, escalation, unknown records, prompt injection, and session memory isolation. The latest verified run passed 6/6 scenarios. Results are saved in `artifacts/agent_evaluation.json`.

3. **Automated regression tests:** The pytest suite validates core functionality and integration behavior. The latest verified run passed 15/15 tests.

All evaluations use deterministic mock LLM behavior. Results demonstrate functionality under the tested conditions, not production-level clinical safety, security, or real-world LLM performance.


## Acceptance evidence

Run `run_all.bat`. The scripts create:
- `dataset_report.json`
- `rag_calibration.json`
- `rag_comparison.json` with per-query arithmetic for both chunkers
- `rag_demo.json`
- `guardrail_demo.json`
- `memory_transcript.json`
- `crew_transcript.json`
- `autogen_review_demo.json`
- `governance_demo.json`
- `cache_demo.json`
- `evaluation_15_queries.json` and `evaluation_summary.json`
- `requests.jsonl` (when API endpoints are called)

The test suite is `python -m pytest -q`.


## Verified execution results

The complete capstone workflow was executed successfully on Windows 11 with Python 3.11.3 in deterministic mock LLM mode.

### Automated validation

- Regression tests: 15/15 passed.
- End-to-end agent evaluation: 6/6 scenarios passed.
- Synthetic dataset: 50 appointment records validated.
- RAG: 12 policy documents indexed using two chunking strategies.
- Calibrated similarity threshold: 0.3081.

### Live FastAPI verification

The following scenarios were manually tested through Swagger UI:

| Scenario | Observed result |
|---|---|
| Cancellation policy | Correct 6-hour policy, grounded response |
| Scheduled appointment | APT-0001, Scheduled, INR 400 |
| Escalation recommendation | APT-0003, Cancelled, INR 1100, escalation recommended |
| Unknown appointment | APT-9999 not verified; no fabricated status |
| Prompt injection | Known instruction-override attempt refused |
| Contact masking | One phone number masked; appointment lookup still worked |

All six live requests returned HTTP 200.

### Audit logging

The contact-masking scenario was checked against `artifacts/requests.jsonl`. The tested phone number was stored as `[REDACTED_CONTACT]`, and the request included a trace ID for audit correlation.

### Limitations

- CrewAI and AutoGen use deterministic mock model responses, not live external LLM inference.
- The review layer uses rule-based evidence checks rather than full semantic entailment verification.
- Policy answers currently prioritize evidence from the highest-ranked document.
- Session memory is in-process and does not provide authenticated user identity.
- Contact masking covers supported phone-number formats, not all sensitive information.
- Prompt-injection testing covers selected attack patterns only.
- The application uses synthetic data and is not intended for clinical decisions or production deployment.


## GitHub submission

Create one **public** GitHub repository, for example `practo-domain-support-agent`. Extract this folder, then:
```bat
git init
git add .
git commit -m "Complete Practo healthcare support agent capstone"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/practo-domain-support-agent.git
git push -u origin main
```
Submit the repository URL. Do not commit `.venv/`, local Chroma persistence, model weights, or secrets; `.gitignore` excludes these. Generated text reports and transcripts are safe to commit.

## Framework notes

CrewAI is run through `.kickoff()` with a custom `crewai.llms.base_llm.BaseLLM` subclass. Tool calls are dispatched from the declared tool argument schema and the generated model text, not by scanning CrewAI's system prompt for the word `Observation:`. The retrieval and lookup tools are separately assigned to their respective agents; only Lookup Agent receives the appointment lookup tool. AutoGen uses `RoundRobinGroupChat(max_turns=2)` and a structured Pydantic verdict. LangChain's `RunnableWithMessageHistory` keeps session history in-process only.
