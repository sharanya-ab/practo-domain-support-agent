"""Runs all capstone demonstrations in a reproducible order."""
import os
import json
import runpy
import truststore
# Use the Windows trust store for HTTPS certificate validation.
truststore.inject_into_ssl()
os.environ["MOCK_LLM"]="true"
os.environ["CREWAI_DISABLE_TELEMETRY"]="true"
os.environ["OTEL_SDK_DISABLED"]="true"
from pathlib import Path
Path("artifacts").mkdir(exist_ok=True)
def stage(label, fn):
    print(f"\n{'='*18} {label} {'='*18}")
    fn()
def main():
    from dataset import APPOINTMENTS, validate_dataset
    def dataset_stage():
        report=validate_dataset(APPOINTMENTS)
        Path("artifacts/dataset_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report,indent=2))
    stage("DATASET",dataset_stage)
    stage("RAG INDEX + CALIBRATION",lambda: __import__("rag.build_index",fromlist=[""]).main() if False else __import__("runpy").run_module("rag.build_index",run_name="__main__"))
    stage("RAG CHUNK COMPARISON",lambda: __import__("runpy").run_module("rag.evaluate_chunking",run_name="__main__"))
    stage("GUARDRAILS + GOVERNANCE",lambda: __import__("runpy").run_module("scripts.demo_guardrails_governance",run_name="__main__"))
    stage("SESSION MEMORY",lambda: __import__("runpy").run_module("scripts.demo_memory",run_name="__main__"))
    stage("CREWAI CREW",lambda: __import__("runpy").run_module("scripts.demo_crew",run_name="__main__"))
    stage("AUTOGEN REVIEW",lambda: __import__("runpy").run_module("scripts.demo_review",run_name="__main__"))
    stage("CACHE",lambda: __import__("runpy").run_module("scripts.demo_cache",run_name="__main__"))
    
    stage(
        "15-QUERY RAG EVALUATION",
        lambda: runpy.run_module(
            "evaluation.run_evaluation",
            run_name="__main__",
        ),
    )

    stage(
        "END-TO-END AGENT EVALUATION",
        lambda: runpy.run_module(
            "evaluation.evaluate_agent",
            run_name="__main__",
        ),
    )

    def test_stage():
        import pytest

        exit_code = pytest.main(["tests/", "-q"])

        if exit_code != 0:
            raise RuntimeError(
                f"Regression tests failed with exit code {exit_code}"
            )

    stage("AUTOMATED REGRESSION TESTS", test_stage)

    print(
        "\nAll configured demonstrations and tests completed successfully."
    )

if __name__=="__main__": main()
