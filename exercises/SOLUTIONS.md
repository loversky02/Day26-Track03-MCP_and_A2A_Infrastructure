# Đáp Án Bài Tập — A2A Multi-Agent Codelab

**⚠️ Khuyến cáo:** Hãy tự làm bài tập trước khi xem đáp án này!

---

## Exercise 2: Tools và Knowledge Base

### TODO 1: Thêm labor law entry

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

### TODO 2: Tạo tool check_statute_of_limitations

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.

    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

### TODO 3: Thêm tool vào danh sách và xử lý

```python
# Thêm vào tools list:
tools = [search_legal_knowledge, check_statute_of_limitations]

# Xử lý tool call:
elif tool_call["name"] == "check_statute_of_limitations":
    tool_result = check_statute_of_limitations.invoke(tool_call["args"])
```

### Giải thích

**Vì sao dùng `@tool` decorator?**
- Tự động tạo schema JSON Schema từ type hints
- LLM hiểu được tham số và mô tả của tool
- LangChain quản lý async execution

**Vì sao phải `.invoke()` thay vì gọi trực tiếp?**
- Tool trong LangChain có thể có callback, tracing
- `.invoke()` đảm bảo tool được execute đúng cách trong LangChain pipeline

---

## Exercise 4: Multi-Agent với Privacy Agent

### TODO 1: Implement privacy_agent

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về bảo vệ dữ liệu cá nhân và GDPR."""
    llm = get_llm()

    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.

Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung: GDPR, data protection, privacy rights, data breach, CCPA."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

### TODO 2: Conditional routing

```python
if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu", "breach", "rò rỉ"]):
    tasks.append(Send("privacy_agent", state))
```

### TODO 3 & 4: Add node + edge

```python
# Trong build_graph():
graph.add_node("privacy_agent", privacy_agent)
# ...
graph.add_edge("privacy_agent", "aggregate_results")
```

### TODO 5: Aggregate results

```python
if state.get("privacy_analysis"):
    sections.append(f"🔒 PHÂN TÍCH BẢO MẬT DỮ LIỆU:\n{state['privacy_analysis']}")
```

### Giải thích

**Vì sao dùng `Send` API?**
- `Send(node_name, state)` cho phép dispatch nhiều nhánh song song
- LangGraph tự động merge kết quả từ các nhánh parallel
- Không cần viết async/await hoặc thread pool

**Vì sao cần `_last_wins` reducer?**
- Khi nhiều node ghi vào cùng một state field (parallel branches)
- `_last_wins` đảm bảo giá trị mới nhất được giữ
- Tránh race condition khi merge state

**Pattern chuẩn cho conditional routing trong LangGraph:**
1. Node routing trả về dict với flags (`needs_tax`, `needs_privacy`)
2. Hàm routing riêng (`route_specialists`) trả về `list[Send]`
3. `graph.add_conditional_edges("check_routing", route_specialists, [...destinations...])`

---

## Bonus Challenge 1: Financial Agent

**File:** `bonus/financial_agent.py` (đã tích hợp vào `exercise_4_multiagent.py`)

```python
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
- CCPA/CPRA: $2,663 unintentional / $7,988 intentional per violation"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"financial_analysis": response.content}
```

**State extension:**
```python
class State(TypedDict):
    # ... existing fields ...
    needs_financial: bool
    financial_analysis: Annotated[str, _last_wins]
```

**Routing:**
```python
needs_financial = any(kw in question_lower for kw in [
    "damage", "damages", "cost", "financial", "revenue", "loss",
    "compensation", "bồi thường", "thiệt hại", "chi phí", "penalty",
    "fine", "payout", "settlement", "insurance", "bảo hiểm",
])

if state.get("needs_financial"):
    tasks.append(Send("financial_agent", state))
```

---

## Bonus Challenge 2: Conversation Memory

**File:** `bonus/conversation_memory.py`

```python
from langgraph.checkpoint.memory import MemorySaver

class MemoryState(TypedDict):
    question: str
    history: Annotated[list, _append]  # append-only reducer
    response: str

def build_memory_graph() -> StateGraph:
    graph = StateGraph(MemoryState)
    graph.add_node("legal_agent", legal_agent_with_memory)
    graph.add_edge(START, "legal_agent")
    graph.add_edge("legal_agent", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# Khi invoke, thêm thread_id để phân biệt conversations:
config = {"configurable": {"thread_id": "user-session-1"}}
result = await graph.ainvoke(initial_state, config)
```

**Key patterns:**
- `MemorySaver` lưu state graph trong memory (có thể thay bằng `SqliteSaver` cho persistence)
- `thread_id` phân biệt các phiên hội thoại
- Reducer `_append` tích lũy history thay vì ghi đè
- Agent prompt tham khảo `state["history"]` để có context từ các lượt trước

---

## Bonus Challenge 3: Retry Logic

**File:** `bonus/retry_logic.py`

3 strategies cho retry:

**Strategy 1: tenacity decorator (recommended)**
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

def is_server_error(exception: BaseException) -> bool:
    """Chỉ retry trên lỗi server (5xx) và network errors."""
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500
    if isinstance(exception, (httpx.ConnectError, httpx.TimeoutException,
                               httpx.RemoteProtocolError)):
        return True
    return False

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(is_server_error),
)
async def delegate_with_retry(endpoint: str) -> str: ...
```

**Strategy 2: Manual retry loop (zero extra dependencies)**
```python
async def delegate_with_manual_retry(endpoint, max_retries=3, base_delay=2.0):
    for attempt in range(max_retries + 1):
        try:
            # ... actual call ...
            return result
        except (httpx.HTTPStatusError, httpx.ConnectError) as e:
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                raise  # 4xx: không retry
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)
    raise last_exc
```

**Strategy 3: Production wrapper cho A2A delegate**
```python
from bonus.retry_logic import delegate_with_retry_wrapper

result = await delegate_with_retry_wrapper(
    delegate, endpoint, question, context_id, trace_id, depth,
    max_retries=3,
)
```

---

## Bonus Challenge 4: Monitoring & Observability

**File:** `bonus/monitoring.py`

**Custom @trace decorator:**
```python
@trace(name="law_agent")
async def law_agent(state: State) -> dict:
    # Tự động log: [law_agent] START → [law_agent] OK (0.50s)
    ...
```

**AgentMetrics collector:**
```python
@dataclass
class AgentMetrics:
    agent_name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    error_count: int = 0
    last_error: str | None = None

    @property
    def avg_time(self) -> float: ...
    @property
    def p95_time(self) -> float: ...
    @property
    def error_rate(self) -> float: ...

metrics = get_metrics("law_agent")
metrics.record(elapsed=0.5)
print(metrics.summary())
# law_agent Metrics:
#   Calls: 5  Avg: 0.200s  P95: 0.301s  Errors: 0 (0.0%)
```

**LangSmith Integration:**
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls_...
export LANGCHAIN_PROJECT=legal-multiagent
```
Tất cả LangChain/LangGraph calls tự động được trace. Dashboard: https://smith.langchain.com

---

## Tổng Kết

| Bài Tập | Kỹ Năng Chính | Độ Khó |
|---|---|---|
| Ex 2: Tools | Function calling, tool binding, knowledge base | ⭐⭐ |
| Ex 4: Privacy Agent | StateGraph, Send API, conditional routing | ⭐⭐⭐ |
| Bonus 1 | State management, new agent creation | ⭐⭐⭐ |
| Bonus 2 | Checkpointing, conversation memory | ⭐⭐⭐⭐ |
| Bonus 3 | Error handling, retry patterns | ⭐⭐⭐⭐ |
| Bonus 4 | Observability, tracing | ⭐⭐⭐ |
