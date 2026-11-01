import re


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def check_draft(draft: str, context: str) -> dict:
    """Conservative offline check for claims supported by context."""
    if not draft.strip() or not context.strip():
        return {
            "approved": False,
            "final_answer": "Please contact the clinic to confirm the policy.",
            "reason": "Draft or supporting context is missing.",
        }
    
    # Reject numerical claims not supported by the evidence.
    number_unit_pattern = (
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:hours?|days?|minutes?|weeks?|months?|"
        r"rupees?|inr|₹|%)\b"
    )

    draft_numbers = set(
        re.findall(number_unit_pattern, draft.lower())
    )
    context_numbers = set(
        re.findall(number_unit_pattern, context.lower())
    )

    if not draft_numbers.issubset(context_numbers):
        return {
            "approved": False,
            "final_answer": (
                "I don't know based on the available clinic policy documents."
            ),
            "reason": "The draft contains numerical claims not supported by the evidence.",
        }


    draft_norm = _normalize(draft)
    context_norm = _normalize(context)

    # Detect potentially risky claims that must be explicitly supported.
    risky_phrases = [
        "full refund",
        "always receive",
        "guaranteed",
        "any time",
        "no charge",
        "free consultation",
    ]

    for phrase in risky_phrases:
        if phrase in draft_norm and phrase not in context_norm:
            return {
                "approved": False,
                "final_answer": "Please contact the clinic to confirm the specific policy.",
                "reason": f"Unsupported claim detected: {phrase}",
            }

    # Allow exact or near-exact wording from the supplied context.
    if draft_norm in context_norm:
        return {
            "approved": True,
            "final_answer": draft,
            "reason": "Draft is directly supported by retrieved context.",
        }

    draft_words = set(re.findall(r"\b\w+\b", draft_norm))
    context_words = set(re.findall(r"\b\w+\b", context_norm))

    # Require strong lexical overlap for the offline demonstration.
    overlap = len(draft_words & context_words) / max(len(draft_words), 1)

    if overlap >= 0.85:
        return {
            "approved": True,
            "final_answer": draft,
            "reason": "Draft has high lexical overlap with retrieved context.",
        }

    return {
        "approved": False,
        "final_answer": "Please contact the clinic to confirm the specific policy.",
        "reason": "Draft is not sufficiently supported by retrieved context.",
    }
