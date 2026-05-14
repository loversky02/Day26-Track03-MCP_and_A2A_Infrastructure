"""Bài Tập 2: Thêm Tools và Knowledge Base

Hoàn thành các TODO để thêm tool và knowledge base entry mới.
Cập nhật thông tin pháp lý mới nhất 2025-2026.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm

# Knowledge base — cập nhật 2025-2026
LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach of contract "
            "include: (1) expectation damages; (2) consequential damages; (3) specific performance; "
            "(4) cover damages. Statute of limitations is typically 4 years (UCC § 2-725). "
            "Note: 2026 rulings (e.g., City of N. Tonawanda v. Penn Power Group, Mar 2026) strictly "
            "apply the 4-year limit for breach of warranty. The Economic Loss Doctrine bars tort "
            "claims for purely economic harm."
        ),
    },
    {
        "id": "labor_law",
        "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination", "việc làm"],
        "text": (
            "Bộ luật Lao động 2019 (Luật số 45/2019/QH14) vẫn là luật khung chính. "
            "Luật Việc làm số 74/2025/QH15 (hiệu lực 1/1/2026) bổ sung: "
            "(1) Mở rộng đối tượng tham gia BHTN — HĐLĐ từ 1 tháng trở lên; "
            "(2) Trần trợ cấp thất nghiệp tối đa 5 lần mức lương tối thiểu vùng; "
            "(3) Rút ngắn thời gian chờ hưởng trợ cấp xuống 10 ngày làm việc; "
            "(4) Người lao động được vay tối đa 200 triệu đồng tạo việc làm "
            "(Nghị định 338/2025/NĐ-CP)."
        ),
    },
    {
        "id": "dtsa_trade_secret",
        "keywords": ["dtsa", "trade secret", "nda", "confidential", "non-disclosure"],
        "text": (
            "Defend Trade Secrets Act (DTSA, 18 U.S.C. § 1836): federal private cause of action. "
            "3-year statute of limitations. Remedies: (1) injunctive relief; (2) actual damages + "
            "unjust enrichment; (3) exemplary damages up to 2x for willful misappropriation; "
            "(4) attorney's fees. Criminal prosecution possible under Economic Espionage Act "
            "(18 U.S.C. § 1832) — penalties up to $5M for individuals / $10M for organizations."
        ),
    },
    {
        "id": "gdpr_overview",
        "keywords": ["gdpr", "data protection", "privacy", "eu", "european"],
        "text": (
            "GDPR fines up to 4% of global annual revenue or EUR 20M (whichever greater). "
            "2025-2026 major fines: MLU/Yango EUR 100M (May 2026, unlawful data transfers to Russia); "
            "Free Mobile EUR 27M (Jan 2026, data security); Reddit EUR 16M (Feb 2026, underage users). "
            "EU-Brazil Mutual Adequacy Decisions (Jan 2026) allow free data flow. "
            "UK DUAA 2025 commenced Feb 2026: PECR fines now £17.5M / 4% turnover."
        ),
    },
    {
        "id": "ccpa_2026",
        "keywords": ["ccpa", "california", "data breach", "privacy", "cpra"],
        "text": (
            "CCPA updated regulations effective Jan 1, 2026: (1) Mandatory cybersecurity audits for "
            "businesses with 50%+ revenue from PI sales or $26.6M+ revenue processing 250K+ consumers; "
            "(2) Privacy Risk Assessments required before processing sensitive PI or using ADMT; "
            "(3) SB 446: data breach notification within 30 days; 500+ affected → notify AG within 15 days; "
            "(4) Penalties: $2,663 per unintentional violation, $7,988 per intentional violation; "
            "(5) Private action: up to $799 per consumer per incident. "
            "Largest fine: Tractor Supply $1.35M (2025). ADMT compliance begins Jan 1, 2027."
        ),
    },
]


@tool
def search_legal_knowledge(query: str) -> str:
    """Tìm kiếm trong knowledge base pháp lý."""
    query_lower = query.lower()
    for entry in LEGAL_KNOWLEDGE:
        if any(kw in query_lower for kw in entry["keywords"]):
            return f"[{entry['id']}] {entry['text']}"
    return "Không tìm thấy thông tin liên quan."


@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án (cập nhật 2026).

    Args:
        case_type: Loại vụ án (contract, tort, property, trade_secret, fbar, fcpa, fraud)
    """
    limits = {
        "contract": (
            "4 năm theo UCC § 2-725 (breach of sale of goods). "
            "2026 rulings: áp dụng nghiêm ngặt, ngoại lệ 'express warranty of future performance'."
        ),
        "tort": (
            "2-3 năm tùy bang. CT Supreme Court 2026: UCC Article 9 statutory damages → "
            "3-year tort statute (Connex Credit Union v. Madgic, Apr 2026)."
        ),
        "property": "5 năm (typical state law).",
        "trade_secret": "3 năm theo DTSA (18 U.S.C. § 1836).",
        "fbar": (
            "6 năm — nhưng clock không chạy nếu chưa từng file FBAR. "
            "2026 penalty: $16,536 non-willful / $165,353+ willful (50% account balance). "
            "Reyes (2nd Cir. Jan 2026): 'reckless disregard' = willful."
        ),
        "fcpa": "5 năm (anti-bribery) / 6 năm (accounting provisions) / 7 năm (money laundering).",
        "fraud": "3-6 năm tùy bang và loại fraud. SEC: 5 năm hoặc 10 năm (28 U.S.C. § 2462).",
    }
    result = limits.get(case_type.lower())
    if result:
        return result
    return (
        f"Không xác định cho '{case_type}'. "
        f"Các loại hỗ trợ: {', '.join(limits.keys())}"
    )


async def main():
    load_dotenv()
    llm = get_llm()

    tools = [search_legal_knowledge, check_statute_of_limitations]
    llm_with_tools = llm.bind_tools(tools)

    question = "Thời hiệu khởi kiện vụ vi phạm hợp đồng là bao lâu?"

    messages = [
        SystemMessage(content="Bạn là chuyên gia pháp lý. Sử dụng tools để tra cứu thông tin."),
        HumanMessage(content=question),
    ]

    print(f"Câu hỏi: {question}\n")

    # First LLM call - decide which tools to use
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    # Execute tools if requested
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"🔧 Gọi tool: {tool_call['name']}")
            tool_result = None

            if tool_call["name"] == "search_legal_knowledge":
                tool_result = search_legal_knowledge.invoke(tool_call["args"])
            elif tool_call["name"] == "check_statute_of_limitations":
                tool_result = check_statute_of_limitations.invoke(tool_call["args"])

            if tool_result:
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

        # Second LLM call - synthesize final answer
        final_response = await llm_with_tools.ainvoke(messages)
        print(f"\n✅ Kết quả:\n{final_response.content}")
    else:
        print(f"\n✅ Kết quả:\n{response.content}")


if __name__ == "__main__":
    asyncio.run(main())
