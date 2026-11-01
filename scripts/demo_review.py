import json
from pathlib import Path
from review.autogen_review import review
def run():
    context="Patients should cancel or reschedule at least 6 hours before the appointment start time."
    approved=review("Patients should cancel or reschedule at least 6 hours before the appointment.",context,False)
    revised=review("Patients may cancel any time and always receive a full refund.",context,True)
    result={"approved_case":approved,"revision_case":revised}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/autogen_review_demo.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
if __name__=="__main__": print(json.dumps(run(),indent=2))
