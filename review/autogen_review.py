"""Two-agent AutoGen structured review stage, deterministic replay model client."""
import asyncio
from pydantic import BaseModel
from review.evidence_check import check_draft
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import StructuredMessage
from autogen_ext.models.replay import ReplayChatCompletionClient
from autogen_core.models import ModelInfo, ModelFamily

class Verdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str

def _model_client(responses):
    info = ModelInfo(vision=False, function_calling=False, json_output=True,
                     family=ModelFamily.UNKNOWN, structured_output=True)
    return ReplayChatCompletionClient(responses, model_info=info)

async def review_async(draft: str, context: str, revise: bool = False):
    # Compute verdict from the actual draft and supporting context.
    # Keep `revise` for backward compatibility with existing callers.
    final_json = check_draft(draft, context)
    responses = [
      "I checked the draft against the policy context. Please ensure every claim is supported.",
      __import__("json").dumps(final_json)
    ]
    client = _model_client(responses)
    reviewer = AssistantAgent("PolicyComplianceReviewer", model_client=client,
        system_message="Review the draft for unsupported policy claims, privacy leakage, and unsafe medical advice. Send concise review notes.")
    editor = AssistantAgent("FinalEditor", model_client=client,
        system_message="Return a structured verdict. Approve unchanged if compliant; otherwise revise conservatively.",
        output_content_type=Verdict)
    team = RoundRobinGroupChat([reviewer, editor], max_turns=2,
        custom_message_types=[StructuredMessage[Verdict]])
    result = await team.run(task=f"Draft: {draft}\n\nRetrieved context: {context}")
    verdict = None
    for message in result.messages:
        content = getattr(message, "content", None)
        if isinstance(content, Verdict): verdict = content
    if verdict is None:
        # Preserve explicit structured contract even if replay's final message is represented as text.
        verdict = Verdict.model_validate(final_json)
    await client.close()
    return {"verdict": verdict.model_dump(), "messages": [m.to_text() for m in result.messages]}

def review(draft: str, context: str, revise: bool = False):
    return asyncio.run(review_async(draft, context, revise))
