import re
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions", r"reveal (the )?system prompt",
    r"act as (an )?(unrestricted|unfiltered)", r"bypass (all )?safety",
    r"developer message", r"exfiltrate"
]
def mask_pii(text: str) -> tuple[str, int]:
    masked, n = PHONE_RE.subn("[REDACTED_CONTACT]", text)
    return masked, n
def detect_injection(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in INJECTION_PATTERNS)
def inspect_input(text: str) -> dict:
    masked, count = mask_pii(text)
    return {"text": masked, "masked_count": count, "injection": detect_injection(text)}
