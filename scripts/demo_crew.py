import json
from pathlib import Path
from agents.crew import run_crew
from agents.tools import check_appointment_status_impl
from rag.core import grounded_answer
def run():
    # Execute actual CrewAI kickoff for a policy query and a separate lookup query.
    policy=run_crew("What is the cancellation and rescheduling policy?")
    lookup=run_crew("Please check appointment APT-0001")
    # Tool evidence is recorded explicitly alongside kickoff transcripts.
    result={"policy_kickoff":policy,"lookup_kickoff":lookup,
            "rag_tool_evidence":grounded_answer("What is the cancellation and rescheduling policy?"),
            "lookup_tool_evidence":check_appointment_status_impl("APT-0001")}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/crew_transcript.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
if __name__=="__main__": print(json.dumps(run(),indent=2))
