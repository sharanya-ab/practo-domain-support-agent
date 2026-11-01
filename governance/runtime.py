import re
from dataclasses import dataclass
from functools import wraps
from rag.core import grounded_answer
CACHE = {}
CACHE_STATS = {"calls": 0, "hits": 0}
def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.casefold()).strip()
def cached_grounded_answer(query: str, strategy="fixed_overlap", threshold=None):
    key = (normalize_query(query), strategy)
    if key in CACHE:
        CACHE_STATS["hits"] += 1
        return CACHE[key]
    CACHE_STATS["calls"] += 1
    result = grounded_answer(query, strategy, threshold)
    CACHE[key] = result
    return result
def estimate_tokens(text: str) -> int:
    # Conservative deterministic approximation: one token per ~3.5 characters.
    return max(1, (len(text) + 3) // 4)
def enforce_budget(text: str, budget: int = 1800):
    estimated = estimate_tokens(text)
    if estimated > budget:
        raise ValueError(f"Request budget exceeded: estimated {estimated} tokens > cap {budget}.")
    return estimated
def assert_lookup_least_autonomy(agent_tool_map: dict):
    for agent, tools in agent_tool_map.items():
        if agent != "Lookup Agent" and "check_appointment_status" in tools:
            raise PermissionError(f"Least-autonomy violation: {agent} has appointment lookup access.")
    assert "check_appointment_status" in agent_tool_map.get("Lookup Agent", [])
    return True
RISK_CLASSIFICATION = {
 "level": "High",
 "justification": "The workflow handles appointment records and may encounter healthcare-related context. Although it is restricted to administrative policy support and synthetic data, the healthcare domain and potential medical data exposure warrant the High category under the supplied rubric. It must not diagnose, triage, prescribe, or replace a clinician."
}
