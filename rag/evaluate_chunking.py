import json
from pathlib import Path
from rag.core import retrieve, doc_level_metrics
QUERIES = [
 ("How do I book an appointment?", ["01_appointment_booking"]),
 ("What is the cancellation and rescheduling window?", ["02_cancellation_rescheduling"]),
 ("What are consultation fees for Cardiology?", ["03_consultation_fees"]),
 ("How do I make an insurance claim?", ["04_insurance_claims"]),
 ("What is the prescription refill policy?", ["05_prescription_refills"]),
 ("How long do routine lab tests take?", ["06_lab_turnaround"]),
 ("Who is eligible for telemedicine?", ["07_telemedicine_eligibility"]),
 ("What should I do in an emergency?", ["08_emergency_protocol"]),
]
if __name__ == "__main__":
    report = {}
    for strategy in ["fixed_overlap", "sentence"]:
        report[strategy] = {}
        for q, relevant in QUERIES:
            report[strategy][q] = doc_level_metrics(retrieve(q, strategy, top_k=5), relevant, k=5)
        vals = list(report[strategy].values())
        report[strategy]["average"] = {
          "precision": sum(x["precision"] for x in vals)/len(vals),
          "recall": sum(x["recall"] for x in vals)/len(vals)
        }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/rag_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Recommendation is generated from these measured means; inspect both values before choosing ACTIVE_COLLECTION.")
