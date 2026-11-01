def groundedness_guard(answer: dict) -> dict:
    if not answer.get("grounded") or not answer.get("retrieved"):
        return {"allowed": False, "answer": "I don't know based on the available clinic policy documents."}
    return {"allowed": True, "answer": answer["answer"]}
