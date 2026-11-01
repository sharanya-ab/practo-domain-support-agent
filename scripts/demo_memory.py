import json
from pathlib import Path
from agents.memory import transcript_demo
if __name__=="__main__":
    data=transcript_demo()
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/memory_transcript.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
    print(json.dumps(data,indent=2))
