import os, json
from pathlib import Path
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
from crewai import Agent, Task, Crew, Process
from agents.mock_llm import PractoMockLLM
from agents.tools import rag_lookup, check_appointment_status
from agents.schemas import SupportResponse

def build_crew():
    llm = PractoMockLLM()
    retrieval = Agent(role="Retrieval Agent", goal="Retrieve clinic policy context and never invent policy.", backstory="A careful policy retrieval specialist.", tools=[rag_lookup], llm=llm, verbose=False)
    lookup = Agent(role="Lookup Agent", goal="Check only synthetic appointment records and report exact fields.", backstory="A restricted record lookup specialist.", tools=[check_appointment_status], llm=llm, verbose=False)
    composer = Agent(role="Response Composer", goal="Compose a concise, safe administrative response from supplied evidence.", backstory="A careful support response writer.", tools=[], llm=llm, verbose=False)
    return retrieval, lookup, composer, llm

def run_crew(query: str) -> dict:
    retrieval, lookup, composer, llm = build_crew()
    record_match = __import__("re").search(r"APT-\d{4}", query, __import__("re").I)
    if record_match:
        task = Task(description=f"Use your appointment lookup tool for this exact request: {query}. Report only returned record fields.", expected_output="A concise appointment status result.", agent=lookup)
    else:
        task = Task(description=f"Use your policy retrieval tool for this exact question: {query}. Use only retrieved context.", expected_output="A concise grounded policy answer.", agent=retrieval)
    compose_task = Task(description="Compose the final administrative support answer from the preceding task output. Do not add facts.", expected_output="A safe final answer.", agent=composer, context=[task])
    crew = Crew(agents=[retrieval, lookup, composer], tasks=[task, compose_task], process=Process.sequential, verbose=False)
    
    result = crew.kickoff()
    raw = str(result)

    # Task 1 contains evidence returned by the Retrieval or Lookup Agent.
    evidence = {}
    if result.tasks_output:
        try:
            evidence = json.loads(result.tasks_output[0].raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            evidence = {}

    # Read the calibrated RAG threshold from the saved artifact.
    source_documents = []

    if evidence.get("grounded", False):
        calibration_path = Path("artifacts/rag_calibration.json")
        calibration = json.loads(
            calibration_path.read_text(encoding="utf-8")
        )
        threshold = calibration["chosen_threshold"]

        source_documents = list(dict.fromkeys(
            item["document_id"]
            for item in evidence.get("retrieved", [])
            if item.get("document_id")
            and item.get("similarity", 0) >= threshold
        ))

    # Use the actual escalation decision returned by the lookup tool.
    escalation_recommended = bool(
        evidence.get("escalation_recommended", False)
    )

    parsed = SupportResponse(
        answer=raw[:4000],
        intent="appointment_status" if record_match else "policy",
        source_documents=source_documents,
        escalation_recommended=escalation_recommended,
    )

    return {
        "response": parsed.model_dump(),
        "crew_output": raw,
        "evidence": evidence,
        "llm_calls": llm.call_count,
    }
