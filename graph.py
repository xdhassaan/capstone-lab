"""
graph.py - Supply Chain Disruption Response Agent (SCDRA) LangGraph

Implements the ReAct (Reason + Act) loop using LangGraph:
    START --> agent --> [router] --> tools --> agent --> ... --> END

Components:
    - State:  TypedDict with annotated messages list
    - Agent Node:  Calls Groq LLM with bound tools
    - Tool Node:   Executes tool calls from the LLM's response
    - Router:      Conditional edge — tool_calls → tools, final answer → END

Deliverable for Lab 3, Tasks 2 & 3.
"""

import json
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from tools import ALL_TOOLS

# ── Load environment variables ───────────────────────────────────────
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════
#  TASK 2: Graph State
# ═══════════════════════════════════════════════════════════════════════

class State(TypedDict):
    """The graph state stores the full conversation history.

    The ``messages`` list uses LangGraph's ``add_messages`` reducer, which
    automatically appends new messages rather than overwriting — preserving
    the complete history of thoughts, tool calls, and results through
    every iteration of the ReAct loop.
    """

    messages: Annotated[list[AnyMessage], add_messages]


# ═══════════════════════════════════════════════════════════════════════
#  LLM Configuration (Groq)
# ═══════════════════════════════════════════════════════════════════════

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Bind all 10 project tools so the LLM can generate structured tool_calls
llm_with_tools = llm.bind_tools(ALL_TOOLS)


# ═══════════════════════════════════════════════════════════════════════
#  System Prompt
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are the Supply Chain Disruption Response Agent (SCDRA).

Your mission is to help procurement teams respond to supply chain disruptions
by systematically gathering information, analyzing impact, and producing
actionable response plans.

## Your Toolbox

GROUNDING (Vector DB):
- search_supplier_docs: Semantic search over supplier qualification documents

DATA RETRIEVAL:
- query_inventory_db: Check inventory levels and open purchase orders
- fetch_disruption_alerts: Get current disruption alerts for a region/category
- load_disruption_history: Find how similar past disruptions were handled
- get_supplier_pricing: Compare pricing and lead times across suppliers
- search_sop_wiki: Find standard operating procedures for disruption types

ANALYSIS:
- calculate_financial_impact: Quantify cost exposure and risk score

PLANNING:
- draft_response_plan: Generate a structured action plan

EXECUTION (world-changing — require explicit user approval):
- send_notification: Send alerts via Slack/email
- update_purchase_order: Modify purchase orders in ERP

## Workflow

Follow these steps for every disruption:
1. Understand the disruption — fetch alerts and check SOPs
2. Assess impact — query inventory and historical data
3. Find alternatives — search supplier docs and get pricing
4. Calculate financial exposure
5. Draft a response plan
6. Present the plan to the user for review
7. NEVER execute send_notification or update_purchase_order without explicit approval

Always think step by step. Use tools to gather real data before drawing conclusions.
"""


# ═══════════════════════════════════════════════════════════════════════
#  TASK 2: Agent Node
# ═══════════════════════════════════════════════════════════════════════

def agent_node(state: State) -> dict:
    """The Agent Node: calls the LLM with conversation history and bound tools.

    The LLM examines the full message history and either:
    - Generates tool_calls (wants to use tools) → router sends to Tool Node
    - Generates a text response (final answer) → router sends to END
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ═══════════════════════════════════════════════════════════════════════
#  TASK 2: Tool Node
# ═══════════════════════════════════════════════════════════════════════

tool_node = ToolNode(ALL_TOOLS)


# ═══════════════════════════════════════════════════════════════════════
#  TASK 3: Conditional Router
# ═══════════════════════════════════════════════════════════════════════

def route_agent_output(state: State) -> str:
    """Router that controls the ReAct loop.

    Checks the LLM's last message:
    - If it contains tool_calls → route to the 'tools' node for execution
    - If it contains only text (final answer) → route to END
    """
    last_message = state["messages"][-1]

    # If the LLM wants to call tools, route to the tool node
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"

    # Otherwise, the LLM has produced a final answer — end the loop
    return END


# ═══════════════════════════════════════════════════════════════════════
#  Graph Construction & Compilation
# ═══════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """Construct and compile the ReAct agent graph.

    Graph topology::

        START --> agent --> [route_agent_output] --> tools --> agent
                                |                               ^
                                +-- (final answer) --> END      |
                                +-- (tool_calls) --> tools -----+
    """
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_agent_output)
    graph.add_edge("tools", "agent")

    # Compile the graph
    return graph.compile()


# Compile on import so `from graph import app` works
app = build_graph()


# ═══════════════════════════════════════════════════════════════════════
#  Main — Test the ReAct Loop
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Supply Chain Disruption Response Agent (SCDRA)")
    print("  ReAct Loop — Powered by LangGraph + Groq")
    print("=" * 60)

    test_query = (
        "We just received notice that our primary supplier TPA-001 in Shenzhen "
        "has had a factory fire and cannot fulfill orders for the next 30 days. "
        "Please assess the impact on our inventory, find alternative suppliers, "
        "calculate the financial exposure, and draft a response plan."
    )

    print(f"\nUser: {test_query}\n")
    print("-" * 60)

    result = app.invoke(
        {"messages": [HumanMessage(content=test_query)]},
        {"recursion_limit": 25},
    )

    # Print the conversation trace
    for msg in result["messages"]:
        role = msg.__class__.__name__.replace("Message", "")

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"\n[{role}] Tool calls:")
            for tc in msg.tool_calls:
                args_str = json.dumps(tc["args"], indent=2) if isinstance(tc.get("args"), dict) else str(tc.get("args", ""))
                print(f"  -> {tc['name']}({args_str[:120]}{'...' if len(args_str) > 120 else ''})")

        if hasattr(msg, "content") and msg.content:
            preview = msg.content[:300]
            if role == "Human":
                print(f"\n[{role}]: {preview}")
            elif role == "AI":
                print(f"\n[{role}]: {preview}{'...' if len(msg.content) > 300 else ''}")
            elif role == "Tool":
                print(f"\n[{role}]: {preview[:150]}{'...' if len(msg.content) > 150 else ''}")

    print("\n" + "=" * 60)
    print("Agent finished.")


