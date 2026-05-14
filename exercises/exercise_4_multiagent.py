"""Bài Tập 4: Thêm Privacy Agent vào Multi-Agent System

Hoàn thành các TODO để thêm privacy agent và conditional routing.
Cập nhật prompts với thông tin pháp lý mới nhất 2025-2026.
"""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.llm import get_llm


def _last_wins(left: str | None, right: str | None) -> str:
    """Reducer: giá trị mới ghi đè giá trị cũ."""
    return right if right is not None else (left or "")


class State(TypedDict):
    question: str
    law_analysis: Annotated[str, _last_wins]
    needs_tax: bool
    needs_compliance: bool
    needs_privacy: bool
    needs_financial: bool
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis: Annotated[str, _last_wins]
    financial_analysis: Annotated[str, _last_wins]
    final_response: str


def law_agent(state: State) -> dict:
    """Agent phân tích pháp lý tổng quát."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia pháp lý. Phân tích câu hỏi sau:

{state['question']}

Tập trung vào: hợp đồng, trách nhiệm dân sự, quyền và nghĩa vụ pháp lý.
Tham khảo các quy định hiện hành 2025-2026 nếu có liên quan."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"law_analysis": response.content}


def check_routing(state: State) -> dict:
    """Quyết định gọi agents nào dựa trên nội dung câu hỏi."""
    question_lower = state["question"].lower()

    needs_tax = any(kw in question_lower for kw in ["tax", "irs", "thuế", "fbar", "fatca", "evasion"])

    needs_compliance = any(kw in question_lower for kw in [
        "compliance", "sec", "regulation", "sox", "fcpa", "aml", "bribery",
        "corruption", "money laundering",
    ])

    needs_privacy = any(kw in question_lower for kw in [
        "data", "privacy", "gdpr", "ccpa", "dữ liệu", "breach", "rò rỉ",
        "personal information", "bảo mật", "cybersecurity", "consent",
    ])

    needs_financial = any(kw in question_lower for kw in [
        "damage", "damages", "cost", "financial", "revenue", "loss",
        "compensation", "bồi thường", "thiệt hại", "chi phí", "penalty",
        "fine", "payout", "settlement", "insurance", "bảo hiểm",
        "liquidated", "fiscal", "monetary",
    ])

    print(f"  [check_routing] tax={needs_tax}, compliance={needs_compliance}, privacy={needs_privacy}, financial={needs_financial}")
    return {
        "needs_tax": needs_tax,
        "needs_compliance": needs_compliance,
        "needs_privacy": needs_privacy,
        "needs_financial": needs_financial,
    }


def route_specialists(state: State) -> list[Send]:
    """Routing function: dispatch Send objects based on routing flags."""
    tasks = []
    if state.get("needs_tax"):
        tasks.append(Send("tax_agent", state))
    if state.get("needs_compliance"):
        tasks.append(Send("compliance_agent", state))
    if state.get("needs_privacy"):
        tasks.append(Send("privacy_agent", state))
    if state.get("needs_financial"):
        tasks.append(Send("financial_agent", state))
    if not tasks:
        tasks.append(Send("aggregate_results", state))
    return tasks


def tax_agent(state: State) -> dict:
    """Agent chuyên về thuế — cập nhật 2026."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia thuế. Phân tích khía cạnh thuế trong câu hỏi:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tham khảo các quy định hiện hành:
- Tax evasion (26 U.S.C. § 7201): felony, up to $100K fine ($500K for corporations), 5 years prison
- FBAR penalties 2026: $16,536 non-willful, $165,353+ willful (50% account balance)
- Reyes (2nd Cir. Jan 2026): reckless disregard = willful FBAR violation
- FATCA Form 8938: $10K initial failure, $50K max, 40% penalty on unreported income
- IRS Voluntary Disclosure Practice proposed reforms (Dec 2025): 20% accuracy penalty instead of 75% fraud penalty"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"tax_analysis": response.content}


def compliance_agent(state: State) -> dict:
    """Agent chuyên về compliance — cập nhật 2026."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia compliance. Phân tích khía cạnh tuân thủ:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tham khảo các quy định hiện hành:
- SEC Enforcement FY2025: 456 actions, $17.9B remedies; FY2026 focus: fraud, insider trading, retail investor protection
- DOJ FCPA Guidelines (June 2025): 4 priorities — cartels, U.S. company fair access, national security, serious misconduct
- New SEC SOX Group (Mar 2026): dedicated enforcement for auditor misconduct
- AML: Canaccord Genuity $80M penalty (Mar 2026) — largest BSA penalty for broker-dealer; 160 unfiled SARs
- FinCEN CDD streamlining (Feb 2026): no longer verify beneficial owners at every new account"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"compliance_analysis": response.content}


def privacy_agent(state: State) -> dict:
    """Agent chuyên về bảo vệ dữ liệu cá nhân — cập nhật 2026."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về GDPR, CCPA, và luật bảo vệ dữ liệu cá nhân.

Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tham khảo các quy định hiện hành 2025-2026:

GDPR (EU):
- Fines up to 4% global revenue or EUR 20M
- Q1 2026: EUR 68.18M total fines (394% increase from Q1 2025)
- Largest 2026: MLU/Yango EUR 100M (unlawful Russia transfers), Free Mobile EUR 27M (data security)
- EU-Brazil Mutual Adequacy Decisions (Jan 2026)

CCPA / CPRA (California):
- New regulations effective Jan 1, 2026: mandatory cybersecurity audits, Privacy Risk Assessments
- SB 446: 30-day data breach notification, AG notice in 15 days for 500+ affected
- Penalties: $2,663 unintentional / $7,988 intentional per violation
- ADMT compliance begins Jan 1, 2027

UK DUAA 2025 (effective Feb 2026):
- PECR fines now £17.5M / 4% turnover (up from £500K)
- Mandatory complaints procedure by June 19, 2026
- Relaxed cookie consent for first-party analytics"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}


def financial_agent(state: State) -> dict:
    """Agent phân tích thiệt hại tài chính và bồi thường — cập nhật 2026."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia tài chính pháp lý (forensic accountant / damages expert).

Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Phân tích các khía cạnh tài chính:
- Thiệt hại thực tế (actual damages), thiệt hại gián tiếp (consequential damages)
- Các khoản phạt (fines, penalties, administrative sanctions)
- Chi phí pháp lý (legal fees, court costs, expert witness fees)
- Tổn thất doanh thu (lost revenue, business interruption)
- Thiệt hại uy tín (reputational damage, brand devaluation)
- Tác động bảo hiểm (insurance coverage, premium impact, exclusions)
- Khả năng thu hồi (recovery probability, settlement range)

Tham khảo các mức phạt 2025-2026:
- GDPR: up to 4% global revenue hoặc EUR 20M (Q1 2026: EUR 68.18M total fines)
- FBAR willful: $165,353+ hoặc 50% account balance (2026 adjusted)
- SEC remedies FY2025: $17.9B total, FY2026: fraud focus
- AML/BSA: Canaccord Genuity $80M penalty (Mar 2026)
- CCPA/CPRA: $2,663 unintentional / $7,988 intentional per violation
- Tax evasion (26 U.S.C. § 7201): $100K individual / $500K corporate, 5 years prison
- FATCA Form 8938: $10K-$50K penalties, 40% on unreported income

Đưa ra ước tính định lượng khi có thể (range thiệt hại, khung phạt áp dụng)."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"financial_analysis": response.content}


def aggregate_results(state: State) -> dict:
    """Tổng hợp kết quả từ tất cả agents."""
    llm = get_llm()

    sections = []
    if state.get("law_analysis"):
        sections.append(f"📋 PHÂN TÍCH PHÁP LÝ:\n{state['law_analysis']}")
    if state.get("tax_analysis"):
        sections.append(f"💰 PHÂN TÍCH THUẾ:\n{state['tax_analysis']}")
    if state.get("compliance_analysis"):
        sections.append(f"✅ PHÂN TÍCH TUÂN THỦ:\n{state['compliance_analysis']}")
    if state.get("privacy_analysis"):
        sections.append(f"🔒 PHÂN TÍCH BẢO MẬT DỮ LIỆU:\n{state['privacy_analysis']}")
    if state.get("financial_analysis"):
        sections.append(f"💲 PHÂN TÍCH TÀI CHÍNH:\n{state['financial_analysis']}")

    combined = "\n\n".join(sections)

    prompt = f"""Tổng hợp các phân tích sau thành một báo cáo pháp lý hoàn chỉnh:

{combined}

Câu hỏi gốc: {state['question']}

Hãy tạo một báo cáo ngắn gọn, có cấu trúc rõ ràng, trích dẫn quy định cụ thể."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_response": response.content}


def build_graph() -> StateGraph:
    """Xây dựng multi-agent graph."""
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("law_agent", law_agent)
    graph.add_node("check_routing", check_routing)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("privacy_agent", privacy_agent)
    graph.add_node("financial_agent", financial_agent)
    graph.add_node("aggregate_results", aggregate_results)

    # Define edges
    graph.add_edge(START, "law_agent")
    graph.add_edge("law_agent", "check_routing")
    graph.add_conditional_edges(
        "check_routing",
        route_specialists,
        ["tax_agent", "compliance_agent", "privacy_agent", "financial_agent", "aggregate_results"],
    )
    graph.add_edge("tax_agent", "aggregate_results")
    graph.add_edge("compliance_agent", "aggregate_results")
    graph.add_edge("privacy_agent", "aggregate_results")
    graph.add_edge("financial_agent", "aggregate_results")
    graph.add_edge("aggregate_results", END)

    return graph.compile()


async def main():
    load_dotenv()

    # Test với câu hỏi có liên quan đến privacy, tax, compliance
    question = "Nếu công ty bị rò rỉ dữ liệu khách hàng, hậu quả pháp lý và thuế là gì?"

    print("=" * 70)
    print("MULTI-AGENT SYSTEM với Privacy Agent (cập nhật 2026)")
    print("=" * 70)
    print(f"\nCâu hỏi: {question}\n")
    print("Đang xử lý qua các agents...\n")

    graph = build_graph()

    result = await graph.ainvoke({
        "question": question,
        "law_analysis": "",
        "needs_tax": False,
        "needs_compliance": False,
        "needs_privacy": False,
        "needs_financial": False,
        "tax_analysis": "",
        "compliance_analysis": "",
        "privacy_analysis": "",
        "financial_analysis": "",
        "final_response": "",
    })

    print("\n" + "=" * 70)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 70)
    print(result["final_response"])
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
