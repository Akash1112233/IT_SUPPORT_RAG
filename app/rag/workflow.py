import logging
from typing import Literal
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None

def llm():
    global _llm
    if _llm is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")
        _llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )
    return _llm

def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]

def route_question(state: AgentState):
    router = llm().with_structured_output(RouteDecision, method="json_mode")
    decision = router.invoke(f"""
You route messages for an enterprise IT support assistant.
Use kb for questions about company IT policies, VPN, password reset, MFA, laptop setup,
software access, security, email, devices, troubleshooting, or technology support.
Use direct only for greetings, thanks, or casual chat that needs no company knowledge.
Question: {state['question']}
Return valid JSON like {{"route":"kb"}}.
""")
    return {"source_used": decision.route, "trace": add_trace(state, f"Router → {decision.route.upper()}")}

def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"


def retrieve_kb(state: AgentState):
    """
    Retrieve candidate chunks and preserve their vector-store scores
    for diagnostics. Scores alone do not establish relevance.
    """
    if not state["current_query"].strip():
        return {
            "kb_docs": [],
            "trace": add_trace(state, "Private KB retrieval skipped → empty query"),
        }

    from app.rag.vectorstore import get_vectorstore

    results = get_vectorstore().similarity_search_with_score(
        query=state["current_query"],
        k=settings.top_k,
    )

    documents = []

    for document, score in results:
        # Keep retrieval diagnostics in metadata without changing
        # the original chunk text.
        document.metadata["_retrieval_score"] = float(score)
        documents.append(document)

    return {
        "kb_docs": documents,
        "trace": add_trace(
            state,
            f"Private KB retrieval → {len(documents)} candidate chunks",
        ),
    }



def grade_kb(state: AgentState):
    """
    Decide whether the retrieved private evidence directly answers
    the user's question, rather than merely sharing similar terms.
    """
    documents = state.get("kb_docs", [])

    if not documents:
        return {
            "kb_grade": "weak",
            "trace": add_trace(
                state,
                "KB evidence grade → WEAK (no documents retrieved)",
            ),
        }

    context_parts = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get(
            "document_name",
            document.metadata.get("source", "unknown"),
        )
        context_parts.append(
            f"Evidence {index}\n"
            f"Source: {source}\n"
            f"Content:\n{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_mode",
    )

    grade = grader.invoke(f"""
You are a strict evidence-relevance evaluator for an enterprise
IT support knowledge base.

User question:
{state['question']}

Retrieved evidence:
{context}

Apply these rules:
1. Grade GOOD only when the retrieved text directly and sufficiently
   supports an answer to the specific question.
2. Grade WEAK when the evidence is unrelated, only broadly related,
   incomplete, or does not contain the requested troubleshooting steps.
3. An IT-related document is not automatically relevant to every IT question.
4. Do not fill missing information using general knowledge.
5. If the question asks about Wi-Fi and the evidence only discusses email
   or software installation, grade WEAK.
6. Consider all retrieved chunks together, but do not assume facts
   that the text does not state.

Return valid JSON with exactly one field:
{{"grade": "good"}}
or
{{"grade": "weak"}}
""")

    result = grade.grade.lower()

    if result not in {"good", "weak"}:
        result = "weak"

    return {
        "kb_grade": result,
        "trace": add_trace(
            state,
            f"KB evidence grade → {result.upper()}",
        ),
    }


def after_kb(
    state: AgentState,
) -> Literal["generate_from_kb", "search_web"]:
    """
    Choose the next graph node based on KB evidence quality.
    """
    if state["kb_grade"] == "good":
        return "generate_from_kb"

    return "search_web"


def search_web(state: AgentState):
    """
    Search the web when private knowledge-base evidence is insufficient.
    """
    result = web_search_tool().invoke({
        "query": state["current_query"]
    })

    lines = []
    citations = []

    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + str(result["answer"]))

        for item in result.get("results", []):
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")

            lines.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

            if url:
                citations.append({
                    "title": title or url,
                    "url": url,
                    "type": "web",
                })
    else:
        lines.append(str(result))

    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(
            state,
            "Web fallback → Tavily search",
        ),
    }

def grade_web(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    grade = grader.invoke(f"""
Question: {state['question']}
Web evidence:\n{state['web_results']}
Return good if the evidence is sufficient and directly relevant; otherwise weak.
Return valid JSON like {{"grade":"good"}}.
""")
    return {"web_grade": grade.grade, "trace": add_trace(state, f"Web evidence grade → {grade.grade.upper()}")}

def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"

def rewrite_query(state: AgentState):
    rewritten = llm().invoke(f"""
Rewrite this IT support question for better private knowledge retrieval and vendor web search.
Preserve intent, add useful technical keywords, do not answer, return only the query.
Question: {state['question']}
""").content.strip()
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }


def generate_from_kb(state: AgentState):
    context = "\n\n".join(
        f"[Source: {d.metadata.get('document_name', d.metadata.get('source', 'Private KB'))}]\n"
        f"{d.page_content}"
        for d in state["kb_docs"]
    )

    if not state["kb_docs"] or not context.strip():
        return {
            "answer": (
                "I couldn't find sufficient evidence in the private "
                "knowledge base to answer confidently. Please contact IT support."
            ),
            "source_used": "insufficient_evidence",
            "citations": [],
            "trace": add_trace(
                state,
                "Answer generation stopped → no private KB evidence",
            ),
        }

    answer = llm().invoke(f"""
You are an enterprise IT support assistant.

STRICT EVIDENCE RULES:
- Answer using ONLY the supplied private knowledge-base evidence.
- Do not add steps, policies, systems, credentials, MFA requirements,
  gateway details, or troubleshooting instructions that are not stated
  in the evidence.
- If the evidence does not specify a detail, say it is not specified.
- Do not use general knowledge to fill gaps.
- Be concise and clearly list supported steps.

Question:
{state['question']}

Private knowledge-base evidence:
{context}
""").content

    citations = []
    seen = set()

    for document in state["kb_docs"]:
        source = document.metadata.get(
            "document_name",
            document.metadata.get("source", "Private KB"),
        )

        if source not in seen:
            seen.add(source)
            citations.append({
                "title": str(source).replace("\\", "/").split("/")[-1],
                "url": "",
                "type": "private_kb",
            })

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(
            state,
            "Answer generation → PRIVATE KB",
        ),
    }


def generate_from_web(state: AgentState):
    answer = llm().invoke(f"""
You are an enterprise IT support copilot. The private company KB was insufficient.
Answer ONLY from the web evidence below. Clearly say this is external web information and may need IT validation before changing company-managed systems.
Question: {state['question']}\n\nWeb evidence:\n{state['web_results']}
""").content
    return {"answer": answer, "source_used": "web_search", "trace": add_trace(state, "Answer generation → WEB SEARCH")}

def direct_answer(state: AgentState):
    answer = llm().invoke(f"Respond briefly and naturally to: {state['question']}").content
    return {"answer": answer, "source_used": "direct", "trace": add_trace(state, "Direct response → no retrieval")}

def insufficient(state: AgentState):
    return {
        "answer": "I couldn't find enough reliable evidence in the company knowledge base or external search to answer confidently. Please contact the IT help desk or provide more details.",
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }

def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()

agent_graph = build_graph()

def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)