"""Session-scoped LangChain memory demonstration using RunnableWithMessageHistory."""
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

_STORE = {}
def get_history(session_id: str):
    if session_id not in _STORE: _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]

def _reply(inputs):
    question = inputs["question"]
    history = inputs.get("history", [])
    if "my reference is" in question.lower():
        answer = "I will use that reference in this conversation."
    elif "what is my reference" in question.lower():
        answer = "Your reference is " + next((m.content.split(":",1)[1].strip() for m in reversed(history)
                    if isinstance(m, HumanMessage) and "my reference is:" in m.content.lower()), "not present in this conversation.")
    else:
        answer = "I can help with clinic policies. Please provide the relevant booking reference when needed."
    return answer

_chain = RunnableLambda(_reply)
_with_history = RunnableWithMessageHistory(
    _chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history",
)
def ask(session_id: str, question: str):
    return _with_history.invoke({"question": question},
        config={"configurable": {"session_id": session_id}})
def transcript_demo():
    a = ask("session-A", "My reference is: APT-0007")
    b = ask("session-A", "What is my reference?")
    c = ask("session-B", "What is my reference?")
    return {"session_A": [a,b], "fresh_session_B": [c],
            "history_A_messages": len(get_history("session-A").messages),
            "history_B_messages": len(get_history("session-B").messages)}
