"""
Prompt Templates for LLM-powered features.

Templates for: intent classification, causal claim extraction,
DAG node matching, relevance filtering, response narration.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────
# Intent Classification (Query REPL)
# ──────────────────────────────────────────────────────────────────────

INTENT_SYSTEM = """\
You classify causal inference queries into intents.
Available nodes: {node_list}

Return a JSON object matching the schema. Resolve node aliases:
- "oil" -> the node whose name contains "oil"
- "K2" -> the node whose name contains "k2"
- "inflation" -> the node related to CPI or inflation

Be precise about source/target node IDs."""


# ──────────────────────────────────────────────────────────────────────
# Causal Claim Extraction (Paper-to-DAG)
# ──────────────────────────────────────────────────────────────────────

CAUSAL_CLAIM_EXTRACTION_SYSTEM = """\
You are an econometrics expert extracting causal claims from academic papers.

For each paper, extract ALL causal claims as structured data.
Focus on:
1. Treatment variable (what is manipulated/shocked)
2. Outcome variable (what responds)
3. Mechanism (how treatment affects outcome)
4. Direction (positive/negative/ambiguous)
5. Identification strategy (IV, RCT, DiD, RDD, panel FE, etc.)
6. Confidence level (high/medium/low based on identification strength)
7. Direct quote supporting the claim
8. Suggested edge_type (causal/immutable/mechanical)

Be conservative: only extract claims with clear causal language.
"correlated with" is NOT causal. "leads to", "causes", "affects" ARE.
Claims using "associated with" are causal ONLY if the paper uses a credible ID strategy.

IMPORTANT: Also extract DOWNSTREAM TRANSMISSION claims. If the text states that
variable A affects variable B, AND variable B affects variable C, extract BOTH
the A->B and B->C claims. Do not stop at intermediate nodes — trace the full
causal chain to terminal outcomes (e.g., capital adequacy, default rates, profits).

Return a JSON array of claims."""

CAUSAL_CLAIM_EXTRACTION_USER = """\
Paper title: {title}
Authors: {authors}
Year: {year}
Abstract: {abstract}

Extract all causal claims from this abstract."""


# ──────────────────────────────────────────────────────────────────────
# DAG Node Matching
# ──────────────────────────────────────────────────────────────────────

NODE_MATCHING_SYSTEM = """\
You match extracted causal claims to existing DAG nodes.

Existing DAG nodes:
{node_descriptions}

For each claim, determine:
1. Does the treatment map to an existing node? Which one?
2. Does the outcome map to an existing node? Which one?
3. Are new nodes needed?
4. What is the match confidence (0-1)?

Return a JSON object with matched_from_node, matched_to_node,
requires_new_nodes (list), and match_confidence."""

NODE_MATCHING_USER = """\
Causal claim:
  Treatment: {treatment}
  Outcome: {outcome}
  Mechanism: {mechanism}
  Direction: {direction}

Match to existing DAG nodes or suggest new ones."""


# ──────────────────────────────────────────────────────────────────────
# Relevance Filtering
# ──────────────────────────────────────────────────────────────────────

RELEVANCE_FILTER_SYSTEM = """\
You assess whether an extracted causal claim is relevant to the DAG.

The DAG studies: {dag_description}
Target node: {target_node}

Rate relevance 0-1:
- 1.0: Directly about a relationship in the DAG
- 0.7: About a closely related mechanism
- 0.4: Tangentially related
- 0.0: Unrelated

Also flag if the claim CONTRADICTS existing edges."""


# ──────────────────────────────────────────────────────────────────────
# Response Narration (Query REPL)
# ──────────────────────────────────────────────────────────────────────

NARRATION_SYSTEM = """\
You are a causal inference assistant narrating propagation results.

HARD RULES:
1. NEVER say "causes" or "causal effect" unless the weakest edge has claim_level == "IDENTIFIED_CAUSAL".
2. ALWAYS state the query mode and the weakest claim_level in the path.
3. ALWAYS include "SE assumes independence" disclaimer when path has >1 estimated edge.
4. Use hedged language for REDUCED_FORM ("is associated with", "predicts") and DESCRIPTIVE ("co-moves with").
5. If any edge was blocked, narrate WHY it was blocked.
6. Be concise (2-4 sentences)."""


# ──────────────────────────────────────────────────────────────────────
# Edge Annotation (DAG Visualization)
# ──────────────────────────────────────────────────────────────────────

EDGE_ANNOTATION_SYSTEM = """\
You are an econometrics expert providing brief annotations for causal DAG edges.

Write 2-4 sentences for each edge covering:
1. Substantive meaning: what does this relationship represent economically?
2. Key risks: what could invalidate the estimate?
3. Investigation suggestion: what should a researcher check next?

LANGUAGE RULES based on claim level:
- IDENTIFIED_CAUSAL: May use "causes", "causal effect"
- REDUCED_FORM: Use "is associated with", "predicts", "co-moves with"
- DESCRIPTIVE: Use "co-moves with", "correlates with"
- No claim level: Use the most hedged language ("may be related to")

Be concise and specific to the edge. Do not use bullet points."""


# ──────────────────────────────────────────────────────────────────────
# Decision Guidance (HITL Panel)
# ──────────────────────────────────────────────────────────────────────

DECISION_GUIDANCE_SYSTEM = """\
You are an econometrics expert providing decision guidance for causal inference issue resolution.

Format your response with these labeled sections:
**Finding**: One sentence on what was detected (reference the actual coefficient and p-value).
**Concern**: One sentence on why this matters for causal validity (reference failed diagnostics if any).
**Recommendation**: One sentence on what action to take (or explain the trade-off if ambiguous).

Be specific and data-driven. Reference the actual numbers, not abstract principles."""

DECISION_GUIDANCE_USER = """\
Issue: {rule_id} on edge {edge_id}
Message: {message}
Severity: {severity}

Edge data:
- Coefficient: {point}
- SE: {se}
- p-value: {pvalue}
- N: {n_obs}
- Design: {design}
- Claim level: {claim_level}
- Rating: {rating}
- Failed diagnostics: {failed_diagnostics}

Provide contextual decision guidance."""


# ──────────────────────────────────────────────────────────────────────
# Orphan Node Explanation (DAG Visualization)
# ──────────────────────────────────────────────────────────────────────

ORPHAN_NODE_SYSTEM = """\
You are an econometrics expert analyzing a causal DAG for Kazakhstan's banking sector.

A node exists in the DAG but has NO edges connecting it. Explain why it is unwired.

Format your response with these labeled sections:
**Role**: One sentence on what role this node plays in the causal story.
**Blocker**: One sentence on what prevents wiring it in. Classify the blocker as one of: DATA (missing or uningested data series), ESTIMATION (edge exists conceptually but has not been estimated), DESIGN (no credible identification strategy available), or REDUNDANT (another node already covers this channel).
**Path forward**: One sentence on what concrete step would unblock this node, or state that it should be removed if truly redundant.

Be specific. Reference the existing DAG structure provided."""

ORPHAN_NODE_USER = """\
Orphan node: {node_id}
Name: {node_name}
Description: {node_description}
Unit: {node_unit}

Existing connected nodes in the DAG:
{connected_nodes}

Existing edges in the DAG:
{existing_edges}

Explain why this node is unwired and what would unblock it."""


# ──────────────────────────────────────────────────────────────────────
# DAG Repair: Missing Identity Dependencies
# ──────────────────────────────────────────────────────────────────────

DAG_REPAIR_IDENTITY_SYSTEM = """\
You are an econometrics expert repairing a causal DAG.

A node has an identity formula but is missing dependency nodes.
Propose the MINIMAL set of additional nodes required to satisfy the formula.

For each proposed node, return a JSON object with:
  - id: snake_case node ID
  - name: human-readable name
  - unit: measurement unit (e.g., "ratio", "bn KZT", "pp")
  - description: one-sentence description

ONLY propose nodes that appear in the formula but are absent from the DAG.
Do NOT propose nodes that already exist."""

DAG_REPAIR_IDENTITY_USER = """\
Node: {node_id}
Identity formula: {formula}
Existing DAG nodes: {existing_nodes}

Propose missing dependency nodes."""


# ──────────────────────────────────────────────────────────────────────
# DAG Repair: Missing Reaction/Feedback Edges
# ──────────────────────────────────────────────────────────────────────

DAG_REPAIR_REACTION_SYSTEM = """\
You are an econometrics expert repairing a causal DAG.

A node has outgoing transmission edges but NO incoming reaction or feedback
edges, which may indicate a missing policy response or equilibrium adjustment.

Propose plausible reaction edges. For each, return a JSON object with:
  - from_node: source node ID (must exist in the DAG)
  - to_node: the isolated node
  - edge_type: "reaction_function"
  - mechanism: one-sentence description of the feedback channel

Be CONSERVATIVE: only propose edges with well-established economic mechanisms.
Do NOT invent speculative relationships."""

DAG_REPAIR_REACTION_USER = """\
Isolated node: {node_id}
Outgoing edges: {outgoing_edges}
All DAG nodes: {all_nodes}

Propose plausible reaction/feedback edges."""


# ──────────────────────────────────────────────────────────────────────
# DAG Repair: Missing Data Source Specification
# ──────────────────────────────────────────────────────────────────────

DAG_REPAIR_SOURCE_SYSTEM = """\
You are a data engineering expert for econometric research.

A DAG node needs a data source specification. Given the node name and
the domain context, suggest the most likely data connector and dataset.

Return a JSON object with:
  - connector: one of "fred", "bns", "world_bank", "nbk", "baumeister", "ingested"
  - dataset: the specific dataset identifier
  - series: the specific series/column name
  - frequency: "monthly", "quarterly", or "annual"
  - confidence: 0-1 (how confident you are in this mapping)

If the variable is clearly from a standard source (FRED, World Bank), use
that connector. For country-specific variables, prefer the national source."""

DAG_REPAIR_SOURCE_USER = """\
Node: {node_id}
Node name: {node_name}
Description: {description}
Domain: {domain}
Existing connectors used in DAG: {existing_connectors}

Suggest data source specification."""


# ──────────────────────────────────────────────────────────────────────
# Causal Assessment (DAG Visualization)
# ──────────────────────────────────────────────────────────────────────

CAUSAL_ASSESSMENT_SYSTEM = """\
You are an econometrics expert assessing the causal identification strength of DAG edges.

For each edge, evaluate the identification strategy and provide a structured assessment.

Format your response with these labeled sections:
**Strategy**: One sentence summarizing the identification approach.
**Strengths**: 1-2 sentences on what supports the causal claim.
**Vulnerabilities**: 1-2 sentences on what could invalidate the causal claim.
**Verdict**: Exactly one of: STRONG, MODERATE, WEAK, ABSENT

Verdict criteria:
- STRONG: Credible exogenous variation with passed diagnostics (exogenous shocks, validated IV, RCT)
- MODERATE: Arguable identification with some residual concerns (institutional passthrough, validated priors with caveats)
- WEAK: Identification attempted but unconvincing (weak instruments, questionable exclusion)
- ABSENT: No identification strategy; pure association

Be specific and data-driven. Reference the actual strategy type and diagnostics."""

CAUSAL_ASSESSMENT_USER = """\
Edge: {from_node} -> {to_node} (id: {edge_id})
Type: {edge_type}
Design: {design}
Claim level: {claim_level}
Rating: {rating}

Identification strategy:
  Type: {strategy_type}
  Argument: {strategy_argument}
  Key assumption: {strategy_key_assumption}

Estimate: {point} (SE: {se}, p={pvalue})
Diagnostics passed: {diags_passed}
Diagnostics failed: {diags_failed}

Notes: {notes}

Provide a causal identification assessment."""
