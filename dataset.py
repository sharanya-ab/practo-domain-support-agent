"""Seeded synthetic appointment generator. No real patient information."""
from __future__ import annotations
import random
from collections import Counter
from pathlib import Path
import json

SEED = 42
N_RECORDS = 50
CATEGORIES = ["General Medicine", "Cardiology", "Dermatology", "Pediatrics", "Orthopedics"]
STATUSES = ["Scheduled", "Completed", "Cancelled", "No-Show", "Rescheduled"]
CATEGORY_WEIGHTS = [0.30, 0.18, 0.18, 0.17, 0.17]
STATUS_WEIGHTS = [0.35, 0.30, 0.12, 0.08, 0.15]
FEE_RANGES = {
    "General Medicine": (400, 700), "Cardiology": (900, 1800),
    "Dermatology": (600, 1200), "Pediatrics": (500, 900),
    "Orthopedics": (800, 1600),
}
FOLLOW_UP_PROBABILITY = 0.20

def generate_appointments(seed: int = SEED, n: int = N_RECORDS) -> list[dict]:
    rng = random.Random(seed)
    # Deterministic weighted draws; a small deterministic coverage prefix guarantees
    # minimum vocabulary coverage without editing individual records after generation.
    categories = [rng.choices(CATEGORIES, CATEGORY_WEIGHTS, k=1)[0] for _ in range(n)]
    statuses = [rng.choices(STATUSES, STATUS_WEIGHTS, k=1)[0] for _ in range(n)]
    for i, value in enumerate(CATEGORIES):
        categories[i] = value
    for i, value in enumerate(STATUSES):
        statuses[i] = value
    records = []
    for i, (category, status) in enumerate(zip(categories, statuses), start=1):
        low, high = FEE_RANGES[category]
        records.append({
            "record_id": f"APT-{i:04d}",
            "category": category,
            "status": status,
            "consultation_fee_inr": rng.randrange(low, high + 1, 50),
            "days_since_created": rng.randint(0, 30),
            "follow_up_required": rng.random() < FOLLOW_UP_PROBABILITY,
        })
    # Follow-up rate is accepted only if within the required band. Seed is fixed and
    # validation below fails loudly if a future edit breaks the deterministic design.
    validate_dataset(records)
    return records

def validate_dataset(records: list[dict]) -> dict:
    assert len(records) >= 40, "Need at least 40 appointments"
    required = {"record_id", "category", "status", "consultation_fee_inr",
                "days_since_created", "follow_up_required"}
    assert all(required.issubset(r) for r in records)
    cat = Counter(r["category"] for r in records)
    status = Counter(r["status"] for r in records)
    assert all(cat[x] >= 3 for x in CATEGORIES), f"Category coverage failed: {cat}"
    assert all(status[x] >= 1 for x in STATUSES), f"Status coverage failed: {status}"
    assert all(0 <= r["days_since_created"] <= 30 for r in records)
    assert all(isinstance(r["follow_up_required"], bool) for r in records)
    pct = 100 * sum(r["follow_up_required"] for r in records) / len(records)
    assert 10 <= pct <= 30, f"Follow-up rate {pct:.1f}% outside 10–30%; change seed/weights"
    assert len({r["record_id"] for r in records}) == len(records)
    return {"records": len(records), "category_counts": dict(cat),
            "status_counts": dict(status), "follow_up_true_percent": round(pct, 2),
            "seed": SEED, "category_weights": dict(zip(CATEGORIES, CATEGORY_WEIGHTS)),
            "status_weights": dict(zip(STATUSES, STATUS_WEIGHTS)),
            "fee_range_inr": [400, 1800]}

APPOINTMENTS = generate_appointments()

def get_appointment(record_id: str) -> dict | None:
    return next((r.copy() for r in APPOINTMENTS if r["record_id"].upper() == record_id.upper()), None)

if __name__ == "__main__":
    report = validate_dataset(APPOINTMENTS)
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
