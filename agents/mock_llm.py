"""CrewAI BaseLLM extension for deterministic offline ReAct-style interactions.

Tool dispatch is schema-driven: inspect each declared tool's JSON argument schema and
emit an Action Input object matching that schema. The generated completion is parsed
from this class's output only; system prompt examples are never treated as observations.
"""
import json, re
from crewai.llms.base_llm import BaseLLM

class PractoMockLLM(BaseLLM):
    def __init__(self, model="practo-mock-v1", temperature=0):
        super().__init__(model=model, temperature=temperature)
        self.call_count = 0

    def _text(self, messages):
        if isinstance(messages, str): return messages
        if isinstance(messages, list):
            return "\n".join(str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in messages)
        return str(messages)

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
        self.call_count += 1
        prompt = self._text(messages)
        lower = prompt.lower()
        # Only inspect actual execution history, not the ReAct
        # formatting examples in CrewAI's system instructions.
        history = prompt.split("Begin!")[-1]

        if "\nObservation:" in history:
            observation = history.rsplit("\nObservation:", 1)[-1]
            observation = observation.split("\nThought:", 1)[0].strip()

            return (
                "Thought: I now have the tool evidence.\n"
                "Final Answer: " + observation
            )
        # Tool selection is based on the declared JSON input schema, not tool-name substrings.
        tool_list = tools or []
        # CrewAI 0.130.0 may expose tools through its ReAct prompt
        # instead of the `tools` argument.
        if not tool_list:
            if "Tool Name: check_appointment_status" in prompt:
                match = re.search(r"APT-\d{4}", prompt, re.I)
                if match:
                    return (
                        "Thought: I need to check the appointment record.\n"
                        "Action: check_appointment_status\n"
                        f"Action Input: {json.dumps({'record_id': match.group(0).upper()})}"
                    )

            if "Tool Name: rag_lookup" in prompt:
                match = re.search(
                    r"Current Task:.*?exact question:\s*(.*?)\.\s*Use only retrieved context",
                    prompt,
                    re.I | re.S,
                )
                query = match.group(1).strip() if match else None
                if query:
                    return (
                        "Thought: I need to retrieve the policy evidence.\n"
                        "Action: rag_lookup\n"
                        f"Action Input: {json.dumps({'query': query})}"
                    )
        for tool_spec in tool_list:
            spec = tool_spec if isinstance(tool_spec, dict) else getattr(tool_spec, "to_dict", lambda: {})()
            function = spec.get("function", spec)
            params = function.get("parameters", {})
            props = params.get("properties", {})
            if "record_id" in props and re.search(r"APT-\d{4}", prompt, re.I):
                rid = re.search(r"APT-\d{4}", prompt, re.I).group(0).upper()
                return f"Action: {function.get('name', 'tool')}\nAction Input: {{\"record_id\": \"{rid}\"}}"
            if "query" in props and any(w in lower for w in ["policy", "appointment", "cancel", "fee", "insurance", "lab", "telemedicine", "refill", "home visit", "second opinion"]):
                q = prompt[-500:].replace('"', "'")
                return f"Action: {function.get('name', 'tool')}\nAction Input: {{\"query\": {json.dumps(q)}}}"
        
        # Compose a response using the preceding CrewAI task's evidence.
        # Only parse the explicit context section provided to the Composer.
        if "You are Response Composer" in prompt:
            marker = "This is the context you're working with:"
            if marker in prompt:
                context = prompt.split(marker, 1)[1].split(
                    "\nBegin!", 1
                )[0].strip()

                try:
                    evidence = json.loads(context)

                    if "grounded" in evidence:
                        if evidence.get("grounded"):
                            answer = evidence.get("answer", "")
                        else:
                            answer = (
                                "I could not find sufficiently reliable "
                                "policy information to answer this question."
                            )

                    elif "found" in evidence:
                        if evidence.get("found"):
                            answer = (
                                f"Appointment {evidence['record_id']} "
                                f"is {evidence['status']}. "
                                f"Consultation fee: "
                                f"₹{evidence['consultation_fee_inr']}."
                            )
                        else:
                            answer = (
                                "The requested appointment reference "
                                "was not found."
                            )
                    else:
                        answer = (
                            "I could not verify the requested information."
                        )

                    return (
                        "Thought: I have composed the response "
                        "using the supplied evidence.\n"
                        "Final Answer: " + answer
                    )

                except (json.JSONDecodeError, KeyError, TypeError):
                    return (
                        "Thought: The supplied evidence could not "
                        "be interpreted safely.\n"
                        "Final Answer: I could not verify the "
                        "requested information."
                    )

        # Composer/final answer is deterministic and intentionally cautious.
        if "json" in lower or "pydantic" in lower or "structured" in lower:
            return json.dumps({"answer": "I can help with clinic booking policies and synthetic appointment status. Please provide a booking reference for a status lookup.", "intent": "general", "source_documents": [], "escalation_recommended": False, "safety_notice": "Administrative support only; not medical advice."})
        return "I can help with clinic policy questions and synthetic appointment status. Please share the appointment reference if you need a status check."

    def supports_function_calling(self):
        return True
