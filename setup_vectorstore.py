"""
setup_vectorstore.py - Populate ChromaDB with sample supplier qualification documents.

Run once before using the agent:
    python setup_vectorstore.py

This creates a persistent ChromaDB collection at ./chroma_db/ containing
supplier profiles, certifications, performance data, and compliance records.
"""

import chromadb
from chromadb.utils import embedding_functions

# ── Sample Supplier Documents ────────────────────────────────────────
SUPPLIER_DOCS = [
    {
        "id": "TPA-001",
        "content": (
            "Supplier: TechParts Asia (TPA-001). Location: Shenzhen, China. "
            "Certifications: ISO 9001:2015, ISO 14001:2015, IATF 16949. "
            "Products: Semiconductor chips (MCU-2200, MCU-3300), ceramic capacitors (CAP-100nF, CAP-220nF). "
            "Lead time: 14-21 days standard, 7 days expedited. MOQ: 5000 units. "
            "On-time delivery rate: 94%. Quality reject rate: 0.3%. "
            "Annual capacity: 2M units. Payment terms: Net 30. "
            "Risk notes: Single-source for MCU-2200 line. Located in typhoon-prone region."
        ),
        "metadata": {"supplier_id": "TPA-001", "region": "Asia", "category": "electronics", "tier": "primary"},
    },
    {
        "id": "ECG-002",
        "content": (
            "Supplier: EuroComponents GmbH (ECG-002). Location: Munich, Germany. "
            "Certifications: ISO 9001:2015, REACH compliant, RoHS compliant, AEO certified. "
            "Products: Precision resistors (RES-10K, RES-47K), inductors (IND-100uH, IND-470uH). "
            "Lead time: 7-10 days. MOQ: 1000 units. "
            "On-time delivery rate: 98%. Quality reject rate: 0.1%. "
            "Annual capacity: 5M units. Payment terms: Net 45. "
            "Risk notes: Premium pricing but highest quality scores in portfolio."
        ),
        "metadata": {"supplier_id": "ECG-002", "region": "Europe", "category": "electronics", "tier": "primary"},
    },
    {
        "id": "ALT-003",
        "content": (
            "Supplier: Pacific Semiconductor Corp (ALT-003). Location: Taipei, Taiwan. "
            "Certifications: ISO 9001:2015, IATF 16949, ISO 45001. "
            "Products: Semiconductor chips (MCU-2200 compatible, MCU-3300 compatible), microprocessors. "
            "Lead time: 10-14 days. MOQ: 2000 units. "
            "On-time delivery rate: 91%. Quality reject rate: 0.5%. "
            "Annual capacity: 3M units. Payment terms: Net 30. "
            "Risk notes: Qualified backup for TPA-001. Slightly higher reject rate but faster lead times."
        ),
        "metadata": {"supplier_id": "ALT-003", "region": "Asia", "category": "electronics", "tier": "backup"},
    },
    {
        "id": "ALT-004",
        "content": (
            "Supplier: Nordic Electronics AB (ALT-004). Location: Stockholm, Sweden. "
            "Certifications: ISO 9001:2015, ISO 14001:2015, REACH, RoHS, Conflict Minerals Free. "
            "Products: Resistors (RES-10K compatible), capacitors (CAP-100nF compatible), connectors. "
            "Lead time: 10-14 days. MOQ: 500 units. "
            "On-time delivery rate: 96%. Quality reject rate: 0.15%. "
            "Annual capacity: 4M units. Payment terms: Net 30. "
            "Risk notes: Strong ESG credentials. Lower MOQ makes them ideal for urgent small orders."
        ),
        "metadata": {"supplier_id": "ALT-004", "region": "Europe", "category": "electronics", "tier": "backup"},
    },
    {
        "id": "MFG-005",
        "content": (
            "Supplier: AmeriChip Manufacturing (MFG-005). Location: Austin, Texas, USA. "
            "Certifications: ISO 9001:2015, ITAR compliant, AS9100D (aerospace grade). "
            "Products: High-reliability MCU chips (MCU-2200-MIL, MCU-3300-MIL), custom ASICs. "
            "Lead time: 21-28 days. MOQ: 1000 units. "
            "On-time delivery rate: 97%. Quality reject rate: 0.05%. "
            "Annual capacity: 500K units. Payment terms: Net 60. "
            "Risk notes: Premium pricing (2x standard). Best for defense/aerospace applications."
        ),
        "metadata": {"supplier_id": "MFG-005", "region": "North America", "category": "electronics", "tier": "specialty"},
    },
    {
        "id": "LOG-006",
        "content": (
            "Supplier: FastFreight Logistics (LOG-006). Location: Singapore. "
            "Certifications: C-TPAT certified, AEO certified, GDP compliant. "
            "Services: Air freight, ocean freight, customs brokerage, warehousing. "
            "Coverage: Asia-Pacific to North America and Europe. "
            "Transit times: Air 3-5 days, Ocean 18-25 days. "
            "On-time delivery rate: 92%. "
            "Risk notes: Primary logistics partner for Asian suppliers. "
            "Expedite surcharge: 40% premium for next-day air freight."
        ),
        "metadata": {"supplier_id": "LOG-006", "region": "Asia", "category": "logistics", "tier": "primary"},
    },
    {
        "id": "LOG-007",
        "content": (
            "Supplier: TransEuro Shipping (LOG-007). Location: Rotterdam, Netherlands. "
            "Certifications: AEO certified, ISO 28000 (supply chain security). "
            "Services: Ocean freight, rail freight, last-mile delivery, bonded warehousing. "
            "Coverage: Europe to Asia and North America. "
            "Transit times: Ocean 12-18 days, Rail 15-20 days. "
            "On-time delivery rate: 95%. "
            "Risk notes: Strong European network. Backup for Asian route disruptions via rail."
        ),
        "metadata": {"supplier_id": "LOG-007", "region": "Europe", "category": "logistics", "tier": "backup"},
    },
    {
        "id": "RAW-008",
        "content": (
            "Supplier: SiliconPure Materials (RAW-008). Location: Busan, South Korea. "
            "Certifications: ISO 9001:2015, SEMI standards compliant. "
            "Products: Silicon wafers (200mm, 300mm), rare earth elements, semiconductor-grade chemicals. "
            "Lead time: 30-45 days. MOQ: 100 wafers. "
            "On-time delivery rate: 88%. Quality reject rate: 0.8%. "
            "Annual capacity: 50K wafers. Payment terms: Net 30. "
            "Risk notes: Long lead times. Geopolitical risk due to regional tensions. "
            "Critical raw material supplier for MCU manufacturing."
        ),
        "metadata": {"supplier_id": "RAW-008", "region": "Asia", "category": "raw_materials", "tier": "primary"},
    },
    {
        "id": "PCK-009",
        "content": (
            "Supplier: SafePack Industries (PCK-009). Location: Dongguan, China. "
            "Certifications: ISO 9001:2015, ISTA 3A certified packaging. "
            "Products: Anti-static packaging, ESD-safe trays, moisture barrier bags, custom foam inserts. "
            "Lead time: 5-7 days. MOQ: 10,000 units. "
            "On-time delivery rate: 96%. Quality reject rate: 0.2%. "
            "Annual capacity: 20M units. Payment terms: Net 15. "
            "Risk notes: Located near TPA-001. Regional disruptions may affect both suppliers simultaneously."
        ),
        "metadata": {"supplier_id": "PCK-009", "region": "Asia", "category": "packaging", "tier": "primary"},
    },
    {
        "id": "QUA-010",
        "content": (
            "Supplier Quality Audit Summary - Q4 2024. "
            "TPA-001: PASSED with minor observations. Corrective action on traceability documentation due Feb 2025. "
            "ECG-002: PASSED with zero findings. Gold-tier supplier status renewed. "
            "ALT-003: PASSED with observations. Recommended for increased order allocation. "
            "ALT-004: PASSED. First audit cycle complete. Approved for production orders. "
            "MFG-005: PASSED. Aerospace-grade facility. Highest scores in portfolio. "
            "RAW-008: CONDITIONAL PASS. Improvement needed in delivery consistency. Review in Q2 2025. "
            "PCK-009: PASSED. No significant findings."
        ),
        "metadata": {"supplier_id": "AUDIT-Q4", "region": "Global", "category": "audit", "tier": "report"},
    },
    {
        "id": "COMP-011",
        "content": (
            "Supplier Compliance Matrix - Updated January 2025. "
            "Trade Compliance: All suppliers cleared for US/EU trade. No OFAC/sanctions matches. "
            "Environmental: TPA-001, ECG-002, ALT-004 have ISO 14001. Others pending. "
            "Conflict Minerals: ALT-004 and MFG-005 certified conflict-free. "
            "Data Privacy: ECG-002 and ALT-004 GDPR compliant. "
            "Cybersecurity: MFG-005 CMMC Level 2 certified. Others have basic security controls. "
            "Insurance: All primary suppliers carry $5M+ product liability insurance."
        ),
        "metadata": {"supplier_id": "COMPLIANCE", "region": "Global", "category": "compliance", "tier": "report"},
    },
    {
        "id": "PERF-012",
        "content": (
            "Supplier Performance Rankings - 2024 Annual Review. "
            "Tier 1 (Excellent): ECG-002 (score: 97/100), MFG-005 (score: 96/100). "
            "Tier 2 (Good): ALT-004 (score: 92/100), TPA-001 (score: 90/100), PCK-009 (score: 89/100). "
            "Tier 3 (Acceptable): ALT-003 (score: 84/100), LOG-006 (score: 83/100), LOG-007 (score: 85/100). "
            "Tier 4 (Needs Improvement): RAW-008 (score: 76/100). "
            "Key factors: On-time delivery (30%), Quality (30%), Cost competitiveness (20%), Responsiveness (20%). "
            "Recommendation: Increase allocation to ECG-002 and ALT-004. Develop RAW-008 improvement plan."
        ),
        "metadata": {"supplier_id": "PERF-2024", "region": "Global", "category": "performance", "tier": "report"},
    },
]


def main():
    """Create and populate the ChromaDB supplier documents collection."""
    print("Setting up ChromaDB vector store...")

    # Use sentence-transformers for local embeddings (no API key needed)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Create persistent client
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection if it exists (for clean re-runs)
    try:
        client.delete_collection(name="supplier_docs")
        print("Deleted existing collection.")
    except Exception:
        pass

    # Create collection
    collection = client.create_collection(
        name="supplier_docs",
        embedding_function=ef,
        metadata={"description": "Supplier qualification documents for SCDRA"},
    )

    # Insert documents
    collection.add(
        documents=[doc["content"] for doc in SUPPLIER_DOCS],
        metadatas=[doc["metadata"] for doc in SUPPLIER_DOCS],
        ids=[doc["id"] for doc in SUPPLIER_DOCS],
    )

    print(f"Inserted {len(SUPPLIER_DOCS)} supplier documents into ChromaDB.")
    print(f"Collection: '{collection.name}' at ./chroma_db/")

    # Verify with a test query
    results = collection.query(
        query_texts=["alternative semiconductor chip supplier"],
        n_results=3,
    )
    print(f"\nVerification query: 'alternative semiconductor chip supplier'")
    for i, (doc_id, doc) in enumerate(zip(results["ids"][0], results["documents"][0])):
        print(f"  [{i+1}] {doc_id}: {doc[:80]}...")

    print("\nVector store setup complete.")


if __name__ == "__main__":
    main()
