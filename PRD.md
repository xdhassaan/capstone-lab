# PRD: Supply Chain Disruption Response Agent (SCDRA)

## Problem Statement

### The Bottleneck

When a supply chain disruption occurs -- such as a key supplier missing a delivery, a shipping port closure, a raw material price spike, or a geopolitical trade restriction -- procurement teams at mid-to-large manufacturers must manually:

1. **Identify Impact**: Determine which SKUs, products, and open purchase orders are affected by cross-referencing ERP data, supplier records, and BOM (Bill of Materials) hierarchies.
2. **Find Alternatives**: Search internal supplier qualification databases and external marketplaces for replacement suppliers who meet compliance and certification requirements.
3. **Evaluate Options**: Compare alternative suppliers on cost, lead time, quality history, and minimum order quantities.
4. **Quantify Exposure**: Estimate the financial impact -- revenue at risk, expedite shipping costs, production line downtime costs.
5. **Act**: Draft and send re-routing instructions, update purchase orders in the ERP, and notify downstream logistics teams.

This process currently takes **4 to 8 hours per disruption event**, involves switching between **5+ disconnected systems** (ERP, email, spreadsheets, supplier portals, news feeds), and is highly error-prone under time pressure. Production line stoppages caused by delayed responses cost **$50K-$500K per hour**.

### Proposed Solution

An **agentic AI system** built on the **LangGraph** framework that autonomously:

- **Perceives** disruption signals from news APIs and internal alerts
- **Reasons** through a multi-step workflow to assess impact, find alternatives, and calculate financial exposure
- **Executes** actions (notifications, PO updates) subject to human-in-the-loop approval

This is fundamentally a **multi-step, multi-tool problem** that cannot be solved with a single LLM prompt because it requires live database queries, vector similarity search over document corpora, external API calls, conditional branching based on disruption type and severity, iterative plan refinement with human feedback, and gated execution of world-changing actions.

---

## User Personas

### Persona 1: Sara -- Senior Procurement Manager

| Attribute | Detail |
|---|---|
| **Role** | Senior Procurement Manager |
| **Organization** | Mid-size electronics manufacturer (~500 employees) |
| **Daily Reality** | Manages 200+ supplier relationships across 3 continents |
| **Pain Point** | Spends 60% of disruption-response time on data gathering across ERP, email, and spreadsheets instead of strategic decision-making |
| **Goal** | Receive a ready-to-approve action plan within 30 minutes of a disruption alert |
| **Tech Comfort** | Comfortable with dashboards, Slack, and email; not a programmer |
| **Interaction** | Reviews agent-generated response plans via Slack/web UI; clicks Approve, Modify, or Reject |

### Persona 2: Ahmed -- VP of Supply Chain Operations

| Attribute | Detail |
|---|---|
| **Role** | Vice President, Supply Chain Operations |
| **Pain Point** | Cannot produce real-time financial exposure numbers when the board or CEO asks during a crisis |
| **Goal** | Automated risk scoring and financial impact summaries for executive reporting |
| **Interaction** | Receives escalation alerts for critical-severity events; reviews weekly trend dashboards |

### Persona 3: Priya -- Logistics Coordinator

| Attribute | Detail |
|---|---|
| **Role** | Logistics Coordinator |
| **Pain Point** | Must manually re-route shipments and notify downstream warehouse teams when upstream suppliers change |
| **Goal** | Receive auto-drafted re-routing instructions she can confirm and forward |
| **Interaction** | Gets notifications with pre-filled shipment change forms |

---

## Success Metrics

| # | Metric | Current Baseline | Target | How Measured |
|---|---|---|---|---|
| 1 | **Mean Time to Response Plan (MTTRP)** | 4-8 hours | < 30 minutes | Timestamp delta: disruption alert to approved action plan |
| 2 | **Affected-SKU Identification Accuracy** | ~70% (manual) | > 95% | Comparison of agent output vs. ERP ground truth audit |
| 3 | **Alternative Supplier Coverage** | 2-3 options manually | 5+ ranked automatically | Count of qualified alternatives per event |
| 4 | **Financial Impact Estimation Error** | +/- 40% | +/- 10% | Variance between agent estimate and actual post-resolution cost |
| 5 | **Human Approval Rate (plan quality)** | N/A | > 80% approved without modification | Ratio of "approve" vs. "modify"/"reject" decisions |
| 6 | **Agent Uptime** | N/A | 99.5% | Infrastructure monitoring |

---

## Tool & Data Inventory

### Knowledge Sources (Perception Layer)

| Source | Type | Description | Access Method |
|---|---|---|---|
| **Inventory & PO Database** | PostgreSQL DB | SKUs, current stock levels, open purchase orders, supplier-SKU mappings | SQL queries via `query_inventory_db()` |
| **Supplier Qualification Docs** | PDF/DOCX corpus | Supplier audit reports, certifications (ISO, compliance), performance scorecards | Vector store (ChromaDB) via `search_supplier_docs()` |
| **Disruption News Feed** | External API | Real-time news and alerts about port closures, geopolitical events, weather disruptions | REST API via `fetch_disruption_alerts()` |
| **Historical Disruption Log** | CSV/JSON | Past disruption events, responses taken, outcomes | Pandas DataFrame via `load_disruption_history()` |
| **Supplier Pricing Catalog** | External API | Current pricing and lead times from supplier portals | REST API via `get_supplier_pricing()` |
| **Company SOPs** | Internal Wiki (Markdown) | Standard operating procedures for each disruption category | RAG retrieval via `search_sop_wiki()` |

### Action Tools (Execution Layer)

| Tool Function | Purpose | Side Effect |
|---|---|---|
| `query_inventory_db(sql)` | Query PostgreSQL for affected SKUs, stock levels, open POs | Read-only |
| `search_supplier_docs(query, top_k)` | Semantic search over supplier qualification vector store | Read-only |
| `fetch_disruption_alerts(region, category)` | Pull real-time disruption signals from news/risk API | Read-only |
| `load_disruption_history(disruption_type)` | Load historical response data for similar events | Read-only |
| `get_supplier_pricing(supplier_id, sku)` | Fetch current price and lead time from supplier API | Read-only |
| `search_sop_wiki(query)` | Retrieve relevant SOP section via RAG | Read-only |
| `calculate_financial_impact(affected_orders, alt_pricing)` | Compute cost delta, revenue-at-risk, expedite fees | Computation only |
| `draft_response_plan(context)` | Use LLM to generate a structured action plan document | Generates artifact |
| `send_notification(channel, message, recipients)` | Send Slack/email notification with the response plan | **World-changing** |
| `update_purchase_order(po_id, new_supplier, new_terms)` | Modify PO in ERP system to reflect re-routing | **World-changing** |

---

## System Architecture (LangGraph)

### State Schema

```python
class SupplyChainState(TypedDict):
    # Input
    disruption_event: dict
    disruption_type: Literal["supplier_failure", "logistics_delay",
                              "quality_recall", "price_spike", "geopolitical"]
    severity: Literal["low", "medium", "high", "critical"]

    # Perception outputs
    affected_skus: List[dict]
    affected_orders: List[dict]
    supplier_docs: List[dict]
    sop_guidance: str
    disruption_history: List[dict]

    # Reasoning outputs
    alternative_suppliers: List[dict]
    financial_impact: dict
    risk_score: float

    # Execution outputs
    response_plan: dict
    human_decision: Optional[str]       # "approve" | "reject" | "modify"
    human_feedback: Optional[str]

    # Control flow
    iteration_count: int
    messages: List[dict]
    error_log: List[str]
```

### Graph Nodes

| Node | Purpose | Tools Called |
|---|---|---|
| `classify_disruption` | Intake raw event, classify type and severity | `fetch_disruption_alerts()`, `search_sop_wiki()` |
| `assess_impact` | Query inventory DB for affected SKUs and open POs | `query_inventory_db()`, `load_disruption_history()` |
| `find_alternatives` | Search supplier docs and pricing for replacement options | `search_supplier_docs()`, `get_supplier_pricing()` |
| `calculate_exposure` | Compute financial impact and risk score | `calculate_financial_impact()` |
| `generate_plan` | Synthesize all data into a structured response plan | `draft_response_plan()` |
| `human_review` | **INTERRUPT** -- pause for human approval | None (waits for human input) |
| `execute_actions` | Send notifications and update purchase orders | `send_notification()`, `update_purchase_order()` |

### Graph Flow

```
START --> classify_disruption
              |
         [severity routing]
              |
              v
        assess_impact
              |
              v
       find_alternatives
           /        \
    [alternatives    [no alternatives
      found]           found]
        |                 |
        v                 |
  calculate_exposure      |
        |                 |
        v                 v
      generate_plan  <--------+
          |                    |
          v                    |
     human_review              |
      /    |     \             |
approve  modify   reject       |
   |    (iter<3)    |          |
   v       +--------+         |
execute_actions               END
   |
   v
  END
```

### Key Design Decisions

1. **Human-in-the-loop**: The `human_review` node uses LangGraph's `interrupt_before` mechanism. The graph pauses execution and waits for human input before any world-changing actions.
2. **Conditional branching**: Three routing points -- severity-based routing after classification, alternatives-found check after supplier search, and human decision routing after review.
3. **Feedback loop**: If the human selects "modify", the graph loops back to `generate_plan` with their feedback incorporated (max 3 iterations to prevent infinite loops).
4. **Tool safety**: Read-only tools are called freely; world-changing tools (`send_notification`, `update_purchase_order`) are only reachable after human approval.
