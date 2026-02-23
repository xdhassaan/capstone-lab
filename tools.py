"""
tools.py - Supply Chain Disruption Response Agent Tools

All tools use the @tool decorator from langchain_core.tools with Pydantic
input validation. Mock data is used for demonstration; in production these
would connect to real databases, APIs, and services.

Deliverable for Lab 3, Task 1: Tool Engineering with Pydantic.
"""

import json
from typing import List

from pydantic import BaseModel, Field
from langchain_core.tools import tool
import chromadb
from chromadb.utils import embedding_functions


# ═══════════════════════════════════════════════════════════════════════
#  MOCK DATA — Simulates real databases, APIs, and services
# ═══════════════════════════════════════════════════════════════════════

INVENTORY_DATA = {
    "SKU-MCU2200": {
        "name": "Microcontroller MCU-2200",
        "stock": 1200,
        "reorder_point": 500,
        "supplier_id": "TPA-001",
        "unit_cost": 4.50,
        "category": "semiconductor",
    },
    "SKU-MCU3300": {
        "name": "Microcontroller MCU-3300",
        "stock": 300,
        "reorder_point": 400,
        "supplier_id": "TPA-001",
        "unit_cost": 6.75,
        "category": "semiconductor",
    },
    "SKU-RES10K": {
        "name": "Resistor 10K Ohm",
        "stock": 50000,
        "reorder_point": 10000,
        "supplier_id": "ECG-002",
        "unit_cost": 0.02,
        "category": "passive",
    },
    "SKU-RES47K": {
        "name": "Resistor 47K Ohm",
        "stock": 35000,
        "reorder_point": 8000,
        "supplier_id": "ECG-002",
        "unit_cost": 0.02,
        "category": "passive",
    },
    "SKU-CAP100": {
        "name": "Capacitor 100nF",
        "stock": 42000,
        "reorder_point": 15000,
        "supplier_id": "TPA-001",
        "unit_cost": 0.05,
        "category": "passive",
    },
    "SKU-IND100": {
        "name": "Inductor 100uH",
        "stock": 18000,
        "reorder_point": 5000,
        "supplier_id": "ECG-002",
        "unit_cost": 0.08,
        "category": "passive",
    },
}

PURCHASE_ORDERS = [
    {
        "po_id": "PO-2024-001",
        "supplier_id": "TPA-001",
        "sku": "SKU-MCU2200",
        "quantity": 10000,
        "status": "open",
        "expected_delivery": "2025-03-15",
        "total_value": 45000.00,
    },
    {
        "po_id": "PO-2024-002",
        "supplier_id": "TPA-001",
        "sku": "SKU-MCU3300",
        "quantity": 5000,
        "status": "open",
        "expected_delivery": "2025-03-20",
        "total_value": 33750.00,
    },
    {
        "po_id": "PO-2024-003",
        "supplier_id": "ECG-002",
        "sku": "SKU-RES10K",
        "quantity": 100000,
        "status": "open",
        "expected_delivery": "2025-03-05",
        "total_value": 2000.00,
    },
    {
        "po_id": "PO-2024-004",
        "supplier_id": "TPA-001",
        "sku": "SKU-CAP100",
        "quantity": 50000,
        "status": "in_transit",
        "expected_delivery": "2025-02-28",
        "total_value": 2500.00,
    },
]

DISRUPTION_HISTORY = [
    {
        "event": "Shenzhen port closure 2023",
        "type": "logistics_delay",
        "duration_days": 14,
        "affected_suppliers": ["TPA-001", "PCK-009"],
        "response": "Rerouted via Hong Kong air freight. Activated ALT-003 for urgent MCU orders.",
        "cost_impact": 125000,
        "resolution_time_hours": 6,
    },
    {
        "event": "TPA-001 quality excursion Q2 2023",
        "type": "quality_recall",
        "duration_days": 21,
        "affected_suppliers": ["TPA-001"],
        "response": "Quarantined affected lots. Shifted 60% allocation to ALT-003 for 3 weeks.",
        "cost_impact": 85000,
        "resolution_time_hours": 4,
    },
    {
        "event": "Semiconductor price spike 2022",
        "type": "price_spike",
        "duration_days": 90,
        "affected_suppliers": ["TPA-001", "ALT-003", "RAW-008"],
        "response": "Locked in 6-month forward contracts. Negotiated volume discounts with MFG-005.",
        "cost_impact": 340000,
        "resolution_time_hours": 48,
    },
    {
        "event": "Taiwan strait tensions 2024",
        "type": "geopolitical",
        "duration_days": 30,
        "affected_suppliers": ["ALT-003"],
        "response": "Pre-positioned 30-day safety stock. Activated European backup suppliers.",
        "cost_impact": 200000,
        "resolution_time_hours": 12,
    },
    {
        "event": "ECG-002 factory fire 2022",
        "type": "supplier_failure",
        "duration_days": 45,
        "affected_suppliers": ["ECG-002"],
        "response": "Switched passive components to ALT-004. Expedited orders via air freight.",
        "cost_impact": 95000,
        "resolution_time_hours": 8,
    },
]

SUPPLIER_PRICING = {
    ("TPA-001", "SKU-MCU2200"): {"price": 4.50, "lead_time_days": 18, "moq": 5000},
    ("TPA-001", "SKU-MCU3300"): {"price": 6.75, "lead_time_days": 18, "moq": 5000},
    ("TPA-001", "SKU-CAP100"): {"price": 0.05, "lead_time_days": 14, "moq": 10000},
    ("ALT-003", "SKU-MCU2200"): {"price": 5.25, "lead_time_days": 12, "moq": 2000},
    ("ALT-003", "SKU-MCU3300"): {"price": 7.80, "lead_time_days": 12, "moq": 2000},
    ("ALT-004", "SKU-RES10K"): {"price": 0.025, "lead_time_days": 10, "moq": 500},
    ("ALT-004", "SKU-CAP100"): {"price": 0.06, "lead_time_days": 12, "moq": 500},
    ("ECG-002", "SKU-RES10K"): {"price": 0.02, "lead_time_days": 8, "moq": 1000},
    ("ECG-002", "SKU-RES47K"): {"price": 0.02, "lead_time_days": 8, "moq": 1000},
    ("ECG-002", "SKU-IND100"): {"price": 0.08, "lead_time_days": 8, "moq": 1000},
    ("MFG-005", "SKU-MCU2200"): {"price": 9.00, "lead_time_days": 25, "moq": 1000},
    ("MFG-005", "SKU-MCU3300"): {"price": 13.50, "lead_time_days": 25, "moq": 1000},
}

SOP_CONTENT = {
    "supplier_failure": (
        "SOP-001: Supplier Failure Response Protocol\n"
        "1. Immediately assess which SKUs and open POs are affected.\n"
        "2. Check current inventory levels against 30-day demand forecast.\n"
        "3. Activate pre-qualified backup suppliers within 4 hours.\n"
        "4. Issue expedited POs to backup suppliers if stock falls below reorder point.\n"
        "5. Notify logistics team of new inbound shipment timelines.\n"
        "6. Escalate to VP Supply Chain if financial exposure exceeds $100K.\n"
        "7. Schedule daily status calls until resolution.\n"
        "8. Document lessons learned within 7 days of resolution."
    ),
    "logistics_delay": (
        "SOP-002: Logistics Delay Response Protocol\n"
        "1. Confirm delay duration and root cause with logistics provider.\n"
        "2. Check if in-transit shipments can be rerouted via alternative ports/carriers.\n"
        "3. Assess production impact based on current inventory runway.\n"
        "4. If delay > 7 days, activate backup logistics provider (LOG-007).\n"
        "5. Consider air freight for critical components (40% surcharge applies).\n"
        "6. Update ERP delivery dates for affected POs.\n"
        "7. Notify production planning of revised timelines."
    ),
    "quality_recall": (
        "SOP-003: Quality Recall Response Protocol\n"
        "1. Quarantine all affected lots immediately.\n"
        "2. Trace affected components through BOM to finished goods.\n"
        "3. Issue supplier corrective action request (SCAR) within 24 hours.\n"
        "4. Source replacement components from backup suppliers.\n"
        "5. Conduct incoming inspection at 100% for next 3 shipments.\n"
        "6. Review supplier scorecard and adjust quality rating."
    ),
    "price_spike": (
        "SOP-004: Price Spike Response Protocol\n"
        "1. Verify price increase validity with supplier.\n"
        "2. Check contractual pricing commitments and long-term agreements.\n"
        "3. Evaluate total cost impact across all affected SKUs.\n"
        "4. Negotiate volume commitments for price protection.\n"
        "5. Explore forward contracts to lock in current pricing.\n"
        "6. If increase > 15%, trigger dual-sourcing evaluation."
    ),
    "geopolitical": (
        "SOP-005: Geopolitical Risk Response Protocol\n"
        "1. Monitor situation via approved intelligence feeds.\n"
        "2. Assess exposure by mapping all suppliers in affected region.\n"
        "3. Pre-position 30-day safety stock for critical components.\n"
        "4. Activate suppliers in unaffected regions.\n"
        "5. Review trade compliance and sanctions implications.\n"
        "6. Escalate to legal team if sanctions are involved."
    ),
}


# ═══════════════════════════════════════════════════════════════════════
#  TOOL 1: search_supplier_docs (GROUNDING TOOL — Vector DB)
# ═══════════════════════════════════════════════════════════════════════

class SearchSupplierDocsInput(BaseModel):
    """Input schema for searching supplier qualification documents."""
    query: str = Field(
        description="Semantic search query about suppliers, certifications, "
        "capabilities, or performance (e.g., 'alternative MCU semiconductor supplier')"
    )
    top_k: int = Field(
        default=3,
        description="Number of top matching documents to return (1-10)",
    )


@tool(args_schema=SearchSupplierDocsInput)
def search_supplier_docs(query: str, top_k: int = 3) -> str:
    """Search the supplier qualification vector database using semantic similarity.

    Use this tool to find information about supplier certifications, capabilities,
    performance history, product offerings, and compliance status. Returns the most
    relevant supplier document excerpts from the ChromaDB vector store.
    """
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="supplier_docs", embedding_function=ef)
    results = collection.query(query_texts=[query], n_results=min(top_k, 10))

    formatted: List[str] = []
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        formatted.append(
            f"[Result {i + 1}] (Supplier: {meta.get('supplier_id', 'N/A')}, "
            f"Region: {meta.get('region', 'N/A')}) {doc}"
        )
    return "\n\n".join(formatted) if formatted else "No matching supplier documents found."


# ═══════════════════════════════════════════════════════════════════════
#  TOOL 2: query_inventory_db
# ═══════════════════════════════════════════════════════════════════════

class QueryInventoryDBInput(BaseModel):
    """Input schema for querying the inventory database."""
    sql: str = Field(
        description="SQL-like query describing what inventory data to retrieve. "
        "Examples: 'SELECT * FROM inventory WHERE supplier_id = TPA-001', "
        "'SELECT * FROM purchase_orders WHERE status = open'"
    )


@tool(args_schema=QueryInventoryDBInput)
def query_inventory_db(sql: str) -> str:
    """Query the inventory and purchase order database for SKUs, stock levels, and open POs.

    Use this tool when you need to find which products are affected by a disruption,
    check current stock levels, or find open purchase orders with a specific supplier.
    Accepts natural-language SQL-like queries interpreted against the inventory dataset.
    """
    sql_lower = sql.lower()
    results = []

    if "purchase_order" in sql_lower or "po" in sql_lower:
        for po in PURCHASE_ORDERS:
            if any(
                val in sql_lower
                for val in [po["supplier_id"].lower(), po["status"].lower(), "all", "*"]
            ):
                results.append(po)
        if not results:
            results = PURCHASE_ORDERS
    else:
        for sku_id, data in INVENTORY_DATA.items():
            if any(
                val in sql_lower
                for val in [data["supplier_id"].lower(), sku_id.lower(), "all", "*"]
            ):
                results.append({"sku": sku_id, **data})
        if not results:
            results = [{"sku": k, **v} for k, v in INVENTORY_DATA.items()]

    return json.dumps(results, indent=2)


# ═══════════════════════════════════════════════════════════════════════
#  TOOL 3: fetch_disruption_alerts
# ═══════════════════════════════════════════════════════════════════════

class FetchDisruptionAlertsInput(BaseModel):
    """Input schema for fetching disruption alerts."""
    region: str = Field(
        description="Geographic region to check (e.g., 'Asia', 'Europe', 'North America', 'Global')"
    )
    category: str = Field(
        description="Disruption category: 'logistics_delay', 'supplier_failure', "
        "'quality_recall', 'price_spike', or 'geopolitical'"
    )


@tool(args_schema=FetchDisruptionAlertsInput)
def fetch_disruption_alerts(region: str, category: str) -> str:
    """Fetch real-time disruption alerts from news and risk monitoring feeds.

    Use this tool to get current disruption information for a specific region
    and category. Returns recent alerts with severity, details, and timestamps.
    """
    alerts = [
        {
            "alert_id": "ALERT-2025-0042",
            "region": region,
            "category": category,
            "headline": f"Major {category.replace('_', ' ')} reported in {region}",
            "severity": "high",
            "source": "Supply Chain Risk Monitor",
            "timestamp": "2025-02-23T14:30:00Z",
            "details": (
                f"Significant {category.replace('_', ' ')} detected affecting "
                f"{region} supply chains. Multiple suppliers in the region reporting "
                f"delays of 2-3 weeks. Recommended action: activate backup suppliers "
                f"and review affected purchase orders immediately."
            ),
        }
    ]
    return json.dumps(alerts, indent=2)


# ═══════════════════════════════════════════════════════════════════════
#  TOOL 4: load_disruption_history
# ═══════════════════════════════════════════════════════════════════════

class LoadDisruptionHistoryInput(BaseModel):
    """Input schema for loading historical disruption data."""
    disruption_type: str = Field(
        description="Type of disruption: 'supplier_failure', 'logistics_delay', "
        "'quality_recall', 'price_spike', or 'geopolitical'"
    )


@tool(args_schema=LoadDisruptionHistoryInput)
def load_disruption_history(disruption_type: str) -> str:
    """Load historical disruption response data for similar past events.

    Use this tool to find how similar disruptions were handled in the past,
    including response strategies, resolution times, and cost impacts. This
    helps inform the current response plan with proven strategies.
    """
    matching = [h for h in DISRUPTION_HISTORY if h["type"] == disruption_type]
    if not matching:
        matching = DISRUPTION_HISTORY
    return json.dumps(matching, indent=2)


# ═══════════════════════════════════════════════════════════════════════
#  TOOL 5: get_supplier_pricing
# ═══════════════════════════════════════════════════════════════════════

class GetSupplierPricingInput(BaseModel):
    """Input schema for fetching supplier pricing."""
    supplier_id: str = Field(
        description="Unique supplier identifier (e.g., 'TPA-001', 'ALT-003', 'MFG-005')"
    )
    sku: str = Field(
        description="SKU / part number to get pricing for (e.g., 'SKU-MCU2200')"
    )


@tool(args_schema=GetSupplierPricingInput)
def get_supplier_pricing(supplier_id: str, sku: str) -> str:
    """Fetch current pricing, lead time, and minimum order quantity from a supplier.

    Use this tool to compare costs between current and alternative suppliers.
    Returns unit price, lead time in days, and minimum order quantity (MOQ).
    """
    key = (supplier_id, sku)
    if key in SUPPLIER_PRICING:
        data = SUPPLIER_PRICING[key]
        return json.dumps({"supplier_id": supplier_id, "sku": sku, **data}, indent=2)
    return json.dumps(
        {"error": f"No pricing found for supplier {supplier_id}, SKU {sku}. "
         "Try a different supplier_id or sku combination."}
    )


# ═══════════════════════════════════════════════════════════════════════
#  TOOL 6: search_sop_wiki
# ═══════════════════════════════════════════════════════════════════════

class SearchSOPWikiInput(BaseModel):
    """Input schema for searching the SOP wiki."""
    query: str = Field(
        description="Search query for standard operating procedures "
        "(e.g., 'supplier failure response', 'logistics delay protocol')"
    )


@tool(args_schema=SearchSOPWikiInput)
def search_sop_wiki(query: str) -> str:
    """Retrieve relevant Standard Operating Procedure sections from the company wiki.

    Use this tool to find official company procedures and guidelines for handling
    specific types of supply chain disruptions. Returns the most relevant SOP.
    """
    query_lower = query.lower()
    for key, content in SOP_CONTENT.items():
        if key in query_lower or any(word in query_lower for word in key.split("_")):
            return content
    return (
        "No matching SOP found for the given query. "
        "General guideline: Escalate to procurement manager within 2 hours "
        "and document the disruption event in the incident tracking system."
    )


# ═══════════════════════════════════════════════════════════════════════
#  TOOL 7: calculate_financial_impact
# ═══════════════════════════════════════════════════════════════════════

class CalculateFinancialImpactInput(BaseModel):
    """Input schema for financial impact calculation."""
    affected_orders: str = Field(
        description="JSON string of affected purchase orders with fields: "
        "po_id, quantity, unit_cost or total_value"
    )
    alt_pricing: str = Field(
        description="JSON string of alternative supplier pricing with fields: "
        "supplier_id, sku, price, lead_time_days"
    )


@tool(args_schema=CalculateFinancialImpactInput)
def calculate_financial_impact(affected_orders: str, alt_pricing: str) -> str:
    """Calculate the financial impact of a supply chain disruption.

    Use this tool AFTER identifying affected orders and alternative suppliers to
    quantify financial exposure: additional procurement costs, expedite fees,
    revenue at risk from production delays, and an overall risk score (0-1).
    """
    try:
        orders = json.loads(affected_orders) if isinstance(affected_orders, str) else affected_orders
        pricing = json.loads(alt_pricing) if isinstance(alt_pricing, str) else alt_pricing
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input. Provide valid JSON strings."})

    total_original = sum(o.get("total_value", 0) for o in orders) if isinstance(orders, list) else 0
    cost_delta = total_original * 0.15
    expedite_fees = total_original * 0.08
    revenue_at_risk = total_original * 2.5
    risk_score = min(0.85, cost_delta / max(total_original, 1))

    impact = {
        "total_original_value": round(total_original, 2),
        "estimated_cost_increase": round(cost_delta, 2),
        "expedite_shipping_fees": round(expedite_fees, 2),
        "revenue_at_risk": round(revenue_at_risk, 2),
        "total_financial_exposure": round(cost_delta + expedite_fees, 2),
        "risk_score": round(risk_score, 2),
        "risk_level": "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low",
    }
    return json.dumps(impact, indent=2)


# ═══════════════════════════════════════════════════════════════════════
#  TOOL 8: draft_response_plan
# ═══════════════════════════════════════════════════════════════════════

class DraftResponsePlanInput(BaseModel):
    """Input schema for drafting a response plan."""
    context: str = Field(
        description="Complete context for plan generation including: disruption "
        "description, affected SKUs, alternative suppliers, financial impact, "
        "and SOP guidance"
    )


@tool(args_schema=DraftResponsePlanInput)
def draft_response_plan(context: str) -> str:
    """Generate a structured disruption response plan from gathered intelligence.

    Use this tool AFTER gathering all necessary information: disruption details,
    affected inventory, alternative suppliers, financial impact, and SOP guidance.
    Produces a structured action plan with prioritized steps for human review.
    """
    plan = {
        "plan_id": "PLAN-2025-0042",
        "status": "draft — pending human approval",
        "generated_by": "SCDRA Agent",
        "context_summary": context[:500] + ("..." if len(context) > 500 else ""),
        "recommended_actions": [
            {
                "priority": 1,
                "action": "Activate backup supplier agreements for affected SKUs",
                "timeline": "Immediate (0-4 hours)",
                "owner": "Procurement Manager",
            },
            {
                "priority": 2,
                "action": "Place expedited orders with qualified alternative suppliers",
                "timeline": "Within 24 hours",
                "owner": "Procurement Manager",
            },
            {
                "priority": 3,
                "action": "Notify downstream logistics and warehouse teams of timeline changes",
                "timeline": "Within 24 hours",
                "owner": "Logistics Coordinator",
            },
            {
                "priority": 4,
                "action": "Update affected purchase orders in ERP system",
                "timeline": "Within 48 hours",
                "owner": "Procurement Manager",
            },
            {
                "priority": 5,
                "action": "Schedule follow-up review and document lessons learned",
                "timeline": "7 days post-resolution",
                "owner": "VP Supply Chain",
            },
        ],
        "estimated_resolution_time": "5-10 business days",
        "requires_human_approval": True,
    }
    return json.dumps(plan, indent=2)


# ═══════════════════════════════════════════════════════════════════════
#  TOOL 9: send_notification (WORLD-CHANGING)
# ═══════════════════════════════════════════════════════════════════════

class SendNotificationInput(BaseModel):
    """Input schema for sending notifications."""
    channel: str = Field(
        description="Notification channel: 'slack', 'email', or 'both'"
    )
    message: str = Field(description="The notification message content to send")
    recipients: str = Field(
        description="Comma-separated list of recipients "
        "(e.g., 'procurement-team, logistics-ops, vp-supply-chain')"
    )


@tool(args_schema=SendNotificationInput)
def send_notification(channel: str, message: str, recipients: str) -> str:
    """Send a notification via Slack or email to specified recipients.

    WARNING: This is a WORLD-CHANGING action. Only use when the user has
    explicitly approved sending notifications. Sends disruption alerts or
    response plans to stakeholders.
    """
    return json.dumps(
        {
            "status": "sent",
            "channel": channel,
            "recipients": [r.strip() for r in recipients.split(",")],
            "message_preview": message[:200] + ("..." if len(message) > 200 else ""),
            "timestamp": "2025-02-23T15:00:00Z",
            "note": "[MOCK] Notification simulated — no actual message sent.",
        },
        indent=2,
    )


# ═══════════════════════════════════════════════════════════════════════
#  TOOL 10: update_purchase_order (WORLD-CHANGING)
# ═══════════════════════════════════════════════════════════════════════

class UpdatePurchaseOrderInput(BaseModel):
    """Input schema for updating a purchase order."""
    po_id: str = Field(description="Purchase order ID to update (e.g., 'PO-2024-001')")
    new_supplier: str = Field(description="New supplier ID to reassign the PO to")
    new_terms: str = Field(
        description="Updated terms as JSON string with fields: price, lead_time, quantity"
    )


@tool(args_schema=UpdatePurchaseOrderInput)
def update_purchase_order(po_id: str, new_supplier: str, new_terms: str) -> str:
    """Update an existing purchase order to re-route to a new supplier.

    WARNING: This is a WORLD-CHANGING action. Only use when the user has
    explicitly approved the purchase order modification. Modifies the PO
    in the ERP system to reflect supplier re-routing.
    """
    return json.dumps(
        {
            "status": "updated",
            "po_id": po_id,
            "previous_supplier": "TPA-001",
            "new_supplier": new_supplier,
            "new_terms": new_terms,
            "timestamp": "2025-02-23T15:05:00Z",
            "note": "[MOCK] PO update simulated — no actual ERP change made.",
        },
        indent=2,
    )


# ═══════════════════════════════════════════════════════════════════════
#  TOOL REGISTRY — All tools for graph binding
# ═══════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    search_supplier_docs,
    query_inventory_db,
    fetch_disruption_alerts,
    load_disruption_history,
    get_supplier_pricing,
    search_sop_wiki,
    calculate_financial_impact,
    draft_response_plan,
    send_notification,
    update_purchase_order,
]
