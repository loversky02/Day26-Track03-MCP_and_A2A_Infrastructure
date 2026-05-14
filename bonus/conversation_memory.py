"""Bonus Challenge 2: Conversation Memory với LangGraph MemorySaver.

Demonstrate checkpointing để agent nhớ các câu hỏi trước đó trong cùng session.
"""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from common.llm import get_llm


def _append(left: list | None, right: list | None) -> list:
    """Reducer: nối danh sách messages (append-only)."""
    left = left or []
    right = right or []
    return left + right


class MemoryState(TypedDict):
    question: str
    history: Annotated[list, _append]
    response: str


def legal_agent_with_memory(state: MemoryState) -> dict:
    """Agent pháp lý có memory — nhớ các câu hỏi trước đó."""
    llm = get_llm()

    # Build conversation context từ history
    history_text = ""
    for msg in state.get("history", []):
        if isinstance(msg, HumanMessage):
            history_text += f"\nNgười dùng đã hỏi trước đó: {msg.content}"
        elif isinstance(msg, AIMessage):
            history_text += f"\nTôi đã trả lời trước đó: {msg.content[:200]}..."
        elif hasattr(msg, "content"):
            history_text += f"\n- {msg.content[:200]}"

    prompt = f"""Bạn là trợ lý pháp lý có bộ nhớ. Trả lời câu hỏi hiện tại, có tham khảo
các câu hỏi trước đó trong cùng phiên để đưa ra câu trả lời nhất quán.

LỊCH SỬ HỘI THOẠI:
{history_text if history_text else "(Đây là câu hỏi đầu tiên trong phiên)"}

CÂU HỎI HIỆN TẠI: {state['question']}

Hãy trả lời ngắn gọn. Nếu câu hỏi hiện tại liên quan đến câu hỏi trước đó,
hãy đề cập đến mối liên hệ đó."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "response": response.content,
        "history": [
            HumanMessage(content=state["question"]),
            AIMessage(content=response.content),
        ],
    }


def build_memory_graph() -> StateGraph:
    """Build graph với MemorySaver checkpointing."""
    graph = StateGraph(MemoryState)

    graph.add_node("legal_agent", legal_agent_with_memory)
    graph.add_edge(START, "legal_agent")
    graph.add_edge("legal_agent", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


async def main():
    load_dotenv()

    graph = build_memory_graph()

    print("=" * 70)
    print("BONUS 2: CONVERSATION MEMORY DEMO")
    print("=" * 70)
    print("\nDemo: Agent nhớ câu hỏi trước đó trong cùng thread_id\n")

    # Session 1: Multi-turn conversation
    thread_id = "legal-session-demo-1"

    questions = [
        "GDPR phạt bao nhiêu tiền cho vi phạm dữ liệu?",
        "Mức phạt đó áp dụng thế nào với công ty nhỏ?",
        "Có cách nào giảm mức phạt không?",
    ]

    for i, q in enumerate(questions, 1):
        config = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke(
            {"question": q, "history": [], "response": ""},
            config,
        )

        print(f"--- Turn {i} ---")
        print(f"Q: {q}")
        print(f"A: {result['response'][:250]}...")
        print()

    # Show checkpoint isolation: different thread = fresh memory
    print("--- Session Isolation Demo ---")
    config_new = {"configurable": {"thread_id": "legal-session-demo-2"}}
    result_new = await graph.ainvoke(
        {"question": "Nhắc lại câu hỏi trước tôi đã hỏi là gì?", "history": [], "response": ""},
        config_new,
    )
    print(f"Thread mới (không có history): {result_new['response'][:200]}...")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY:")
    print("  - MemorySaver lưu state graph vào memory (có thể thay bằng SqliteSaver)")
    print("  - thread_id phân biệt các phiên hội thoại khác nhau")
    print("  - Reducer _append giúp tích lũy history thay vì ghi đè")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
