import json
from pathlib import Path
from rag.core import build_indexes, calibrate_threshold, grounded_answer
if __name__ == "__main__":
    Path("artifacts").mkdir(exist_ok=True)
    print("Indexing both independent Chroma collections...")
    print(json.dumps(build_indexes(), indent=2))
    threshold, report = calibrate_threshold()
    print(f"Empirically calibrated threshold: {threshold:.4f}")
    queries = [
      "How do I book an appointment?", "What is the cancellation window?",
      "What are the consultation fees for Cardiology?", "How do insurance claims work?",
      "How long do lab tests take?", "Can I ask who won the cricket world cup?"
    ]
    demo = {q: grounded_answer(q, threshold=threshold) for q in queries}
    Path("artifacts/rag_demo.json").write_text(json.dumps(demo, indent=2), encoding="utf-8")
