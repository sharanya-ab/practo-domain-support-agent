import re
import os, json, time, uuid, logging
from pathlib import Path
from agents.crew import run_crew
from review.autogen_review import review
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field
from rag.core import grounded_answer
from guardrails.input import inspect_input
from guardrails.output import groundedness_guard
from governance.runtime import cached_grounded_answer, enforce_budget
from agents.tools import check_appointment_status_impl
from agents.memory import ask

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY","true")
os.environ.setdefault("OTEL_SDK_DISABLED","true")
app=FastAPI(title="Practo Domain Support Agent", version="1.0.0")
LOG_PATH=Path("artifacts/requests.jsonl")
LOG_PATH.parent.mkdir(exist_ok=True)
logger=logging.getLogger("practo.audit")
logger.setLevel(logging.INFO)

class AskRequest(BaseModel):
    query: str = Field(min_length=2, max_length=5000)
    session_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
    )
class AskResponse(BaseModel):
    trace_id: str
    answer: str
    grounded: bool
    masked_contacts: int
    escalation_recommended: bool = False
class AddDocumentRequest(BaseModel):
    document_id: str
    content: str = Field(min_length=20, max_length=10000)

def audit(trace_id, masked_query, start, status):
    entry={"trace_id":trace_id,"timestamp":time.time(),"duration_ms":round((time.perf_counter()-start)*1000,2),
           "request_text":masked_query,"status":status}
    with LOG_PATH.open("a",encoding="utf-8") as f: f.write(json.dumps(entry,ensure_ascii=False)+"\n")

@app.get("/health")
def health(): return {"status":"ok","mock_llm":True}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest):
    trace = str(uuid.uuid4())
    start = time.perf_counter()
    guard = inspect_input(payload.query)
    escalation_recommended = False

    try:
        if guard["injection"]:
            answer = (
                "I can't follow instructions that attempt "
                "to override system or safety rules."
            )
            grounded = False

        
        else:
            enforce_budget(
                guard["text"],
                int(os.getenv("REQUEST_TOKEN_BUDGET", "1800")),
            )

            query = guard["text"]
            session_id = payload.session_id

            is_reference_statement = bool(
                re.fullmatch(
                    r"\s*my reference is:\s*APT-\d{4}\s*",
                    query,
                    flags=re.IGNORECASE,
                )
            )

            is_reference_question = (
                query.strip().lower() == "what is my reference?"
            )

            if (
                session_id is not None
                and (is_reference_statement or is_reference_question)
            ):
                # Session-scoped LangChain memory demonstration.
                answer = ask(session_id, query)
                grounded = False

            else:
                # Stage 1: CrewAI agents
                crew_result = run_crew(query)
                response = crew_result["response"]
                evidence = crew_result.get("evidence", {})

                escalation_recommended = bool(
                    response.get("escalation_recommended", False)
                )

                draft = response["answer"]
                intent = response["intent"]

                # Stage 2: Prepare reviewer evidence
                if intent == "policy":
                    context = (
                        evidence.get("answer", "")
                        if evidence.get("grounded", False)
                        else ""
                    )
                else:
                    if evidence.get("found", False):
                        context = (
                            f"Appointment {evidence['record_id']} is "
                            f"{evidence['status']}. "
                            f"Consultation fee: "
                            f"₹{evidence['consultation_fee_inr']}."
                        )
                    else:
                        context = ""

                # Stage 3: AutoGen review
                reviewed = review(draft, context)
                verdict = reviewed["verdict"]

                # Stage 4: Evidence validation
                evidence_available = (
                    bool(evidence.get("grounded", False))
                    if intent == "policy"
                    else bool(evidence.get("found", False))
                )

                grounded = bool(
                    evidence_available and verdict["approved"]
                )

                if grounded:
                    answer = verdict["final_answer"]
                elif intent == "appointment_status":
                    answer = (
                        "I couldn't verify that appointment "
                        "from the available records."
                    )
                else:
                    answer = (
                        "I don't know based on the available "
                        "clinic policy documents."
                    )

        audit(trace, guard["text"], start, "ok")

        return AskResponse(
            trace_id=trace,
            answer=answer,
            grounded=grounded,
            masked_contacts=guard["masked_count"],
            escalation_recommended=(
                escalation_recommended if grounded else False
            ),
        )

    except ValueError as e:
        audit(trace, guard["text"], start, "rejected")
        raise HTTPException(status_code=413, detail=str(e))


@app.post("/add-document")
def add_document(payload: AddDocumentRequest):
    # Persist a new text file then rebuild both Chroma collections on next index run.
    target=Path("knowledge_base")/(payload.document_id.replace("..","").replace("/","_")+".txt")
    target.write_text(payload.content,encoding="utf-8")
    return {"added":True,"document_id":payload.document_id,
            "note":"Run python -m rag.build_index to rebuild both indexes."}

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    session_id="ws-"+str(uuid.uuid4())
    try:
        while True:
            message=await ws.receive_text()
            guard=inspect_input(message)
            if guard["injection"]:
                answer="I can't follow instructions that attempt to override system or safety rules."
            else:
                try:
                    enforce_budget(guard["text"],int(os.getenv("REQUEST_TOKEN_BUDGET","1800")))
                    result=cached_grounded_answer(guard["text"])
                    answer=groundedness_guard(result)["answer"]
                except ValueError as exc: answer=str(exc)
            await ws.send_json({"answer":answer,"masked_contacts":guard["masked_count"]})
    except WebSocketDisconnect:
        # Client disconnect is expected; no global state or server shutdown.
        return
