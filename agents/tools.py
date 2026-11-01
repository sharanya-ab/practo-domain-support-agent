import json
from crewai.tools import tool
from dataset import get_appointment, APPOINTMENTS
from rag.core import grounded_answer
from pathlib import Path
def escalation_threshold():
    vals = sorted(0.60*int(r["follow_up_required"]) + 0.40*(1-r["days_since_created"]/30) for r in APPOINTMENTS)
    return vals[int(0.80*(len(vals)-1))]
def check_appointment_status_impl(record_id: str) -> dict:
    row = get_appointment(record_id)
    if not row: return {"found": False, "record_id": record_id}
    score = 0.60*int(row["follow_up_required"]) + 0.40*(1-row["days_since_created"]/30)
    return {"found": True, "record_id": row["record_id"], "status": row["status"],
            "consultation_fee_inr": row["consultation_fee_inr"], "escalation_score": round(score, 4),
            "escalation_threshold": round(escalation_threshold(), 4),
            "escalation_recommended": score >= escalation_threshold()}
@tool("rag_lookup")
def rag_lookup(query: str) -> str:
    """Retrieve clinic-policy knowledge and return grounded policy excerpts for a query."""
    return json.dumps(grounded_answer(query), ensure_ascii=False)
@tool("check_appointment_status")
def check_appointment_status(record_id: str) -> str:
    """Look up a synthetic appointment by record_id; return status, fee, and escalation score."""
    return json.dumps(check_appointment_status_impl(record_id), ensure_ascii=False)
