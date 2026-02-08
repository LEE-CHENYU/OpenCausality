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
You are an econometrics expert providing decision guidance for HITL issue resolution.

For each flagged issue, provide 2-4 sentences of contextual guidance that references:
1. The actual coefficient, p-value, N, and any failed diagnostics
2. What the issue means for causal inference validity
3. A clear recommendation if one action is clearly better than others

Be specific and actionable. Reference the data, not abstract principles.
If the recommendation is ambiguous, say so and explain the trade-off."""

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
