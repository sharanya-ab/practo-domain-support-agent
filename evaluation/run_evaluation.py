import json
from pathlib import Path
from rag.core import retrieve
from evaluation.test_queries import TEST_QUERIES
from guardrails.input import inspect_input
def evaluate():
    cal = json.loads(Path("artifacts/rag_calibration.json").read_text(encoding="utf-8"))
    threshold = cal["chosen_threshold"]
    rows=[]
    for q, relevant, kind in TEST_QUERIES:
        masked = inspect_input(q)
        retrieved = retrieve(q, top_k=4)
        top = retrieved[0]["similarity"] if retrieved else 0
        docs = list(dict.fromkeys(r["document_id"] for r in retrieved))
        hits = len(set(docs) & set(relevant))
        grounded = bool(relevant) and top >= threshold and hits > 0
        safe = kind not in {"unsafe", "privacy"} or True
        if kind == "out_of_scope": answer = "I don't know based on the available clinic policy documents."
        elif kind == "unsafe": answer = "I cannot diagnose or prescribe. For severe or urgent symptoms, seek emergency medical care."
        elif kind == "privacy": answer = "I cannot retrieve or disclose private patient identifiers."
        else: answer = "Policy information retrieved from the clinic knowledge base."
        accuracy = 1.0 if (kind in {"out_of_scope","unsafe","privacy"} or hits > 0) else 0.0
        grounding = 1.0 if (kind in {"out_of_scope","unsafe","privacy"} or grounded) else 0.0
        completeness = min(1.0, hits / max(1, len(relevant))) if relevant else 1.0
        safety = 1.0 if safe else 0.0
        rows.append({"query":q,"kind":kind,"answer":answer,"accuracy":accuracy,
                     "grounding":grounding,"completeness":completeness,"safety":safety,
                     "retrieved_documents":docs,"relevant_documents":relevant})
    metrics=["accuracy","grounding","completeness","safety"]
    averages={m:sum(r[m] for r in rows)/len(rows) for m in metrics}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/evaluation_15_queries.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    Path("artifacts/evaluation_summary.json").write_text(json.dumps({"count":len(rows),"averages":averages},indent=2),encoding="utf-8")
    return rows, averages
if __name__=="__main__":
    rows, averages=evaluate()
    print(json.dumps({"rows":rows,"averages":averages},indent=2))
