#!/usr/bin/env python3
"""
Propagation result validator for PostToolUse hooks.

Reads a PostToolUse JSON payload from stdin, validates the MCP propagation
result for structural issues that could lead to misinterpretation.

This is a defense-in-depth layer — the PropagationEngine already has 7
guardrails. This script catches leaks (e.g. a diagnostic_only edge on an
unblocked path would be an engine bug).

Exit codes:
  0 — No errors found (or non-applicable tool)

Output (only when issues found):
  {"decision": "block", "reason": "..."}   for critical errors
  — or prints warning text to stderr (exit 2) for non-critical warnings

Stdlib-only. No external dependencies.
"""

import json
import sys

# --------------------------------------------------------------------------
# Unit compatibility matrix (mirrors UnitSpec.compatible_with in propagation.py)
# --------------------------------------------------------------------------

_COMPATIBLE_PAIRS = {
    frozenset({"pct", "log_point"}),
    frozenset({"pct", "growth"}),
    frozenset({"growth", "log_point"}),
}


def _units_compatible(u1: str, u2: str) -> bool:
    """Check if two unit kinds are compatible for chain propagation."""
    if u1 == u2:
        return True
    return frozenset({u1, u2}) in _COMPATIBLE_PAIRS


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_reaction_function_leak(paths: list[dict]) -> str | None:
    """Scan unblocked paths for diagnostic_only edges — should never happen."""
    for i, path in enumerate(paths):
        if path.get("is_blocked"):
            continue
        for edge in path.get("edges", []):
            role = edge.get("role", "")
            if role == "diagnostic_only":
                return (
                    f"CRITICAL: Path {i} has diagnostic_only edge "
                    f"'{edge.get('edge_id', '?')}' on an unblocked path. "
                    f"This is a propagation engine bug — reaction functions "
                    f"must never appear on open paths."
                )
    return None


def check_all_blocked(paths: list[dict], tool_name: str) -> str | None:
    """Warn if every path is blocked — Claude should not silently ignore this."""
    if not paths:
        return None
    open_paths = [p for p in paths if not p.get("is_blocked")]
    if len(open_paths) == 0:
        blocked_reasons = set()
        for p in paths:
            for reason in p.get("blocked_reasons", []):
                blocked_reasons.add(reason)
        reason_str = "; ".join(sorted(blocked_reasons)) if blocked_reasons else "unknown"
        return (
            f"All {len(paths)} propagation path(s) are blocked. "
            f"Reasons: {reason_str}. "
            f"There is no propagatable path for this query — "
            f"do not present estimated effects."
        )
    return None


def check_unit_chain(paths: list[dict]) -> str | None:
    """Verify consecutive edges have compatible treatment/outcome units."""
    for i, path in enumerate(paths):
        if path.get("is_blocked"):
            continue
        edges = path.get("edges", [])
        for j in range(len(edges) - 1):
            # Outcome unit of edge j should be compatible with treatment unit of edge j+1
            # The MCP result only includes treatment_unit per edge, but we can check
            # consecutive treatment units for obvious mismatches
            curr_unit = edges[j].get("treatment_unit", "")
            next_unit = edges[j + 1].get("treatment_unit", "")
            if curr_unit and next_unit and not _units_compatible(curr_unit, next_unit):
                return (
                    f"Unit chain break on path {i}: edge "
                    f"'{edges[j].get('edge_id', '?')}' has treatment_unit "
                    f"'{curr_unit}' but next edge "
                    f"'{edges[j+1].get('edge_id', '?')}' has treatment_unit "
                    f"'{next_unit}'. These may not be compatible."
                )
    return None


def check_magnitude_plausibility(result: dict) -> str | None:
    """Flag if scaled_effect exceeds 10x the input magnitude (suggests unit mismatch)."""
    magnitude = result.get("magnitude")
    if magnitude is None or magnitude == 0:
        return None
    abs_mag = abs(magnitude)

    for i, path in enumerate(result.get("paths", [])):
        if path.get("is_blocked"):
            continue
        scaled = path.get("scaled_effect")
        if scaled is None:
            continue
        if isinstance(scaled, str):  # "NaN", "Inf"
            return (
                f"Path {i} has non-finite scaled_effect: {scaled}. "
                f"This suggests a data or estimation issue."
            )
        if abs(scaled) > abs_mag * 10:
            return (
                f"Path {i} scaled_effect ({scaled:.4g}) exceeds 10x the "
                f"input magnitude ({magnitude}). This may indicate a unit "
                f"mismatch between the shock unit and edge coefficient units."
            )
    return None


def check_blocked_reasons_populated(paths: list[dict]) -> str | None:
    """Verify that blocked paths actually have blocked_reasons populated."""
    for i, path in enumerate(paths):
        if path.get("is_blocked") and not path.get("blocked_reasons"):
            return (
                f"Path {i} is marked as blocked but has no blocked_reasons. "
                f"Claude must be able to explain why paths are blocked."
            )
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_name = data.get("tool_name", "")

    # Only run full checks on propagation tools
    is_propagation = any(
        t in tool_name
        for t in ("propagate_shock", "propagate_policy")
    )
    is_path_tool = "find_paths" in tool_name

    if not is_propagation and not is_path_tool:
        sys.exit(0)

    # Parse the tool_response — it's a JSON string from MCP
    raw_response = data.get("tool_response", "")
    if isinstance(raw_response, str):
        try:
            result = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError):
            # If it's not valid JSON, try extracting from TextContent
            sys.exit(0)
    elif isinstance(raw_response, dict):
        result = raw_response
    elif isinstance(raw_response, list):
        # MCP returns list of TextContent objects
        texts = []
        for item in raw_response:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            try:
                result = json.loads(texts[0])
            except (json.JSONDecodeError, TypeError):
                sys.exit(0)
        else:
            sys.exit(0)
    else:
        sys.exit(0)

    # Skip if the result itself is an error
    if "error" in result:
        sys.exit(0)

    paths = result.get("paths", [])

    # --- Critical checks (block) ---
    critical = check_reaction_function_leak(paths)
    if critical:
        json.dump({"decision": "block", "reason": critical}, sys.stdout)
        sys.stdout.write("\n")
        sys.exit(0)

    # --- Warning checks (stderr, exit 2) ---
    warnings = []

    blocked_warning = check_all_blocked(paths, tool_name)
    if blocked_warning:
        warnings.append(blocked_warning)

    unit_warning = check_unit_chain(paths)
    if unit_warning:
        warnings.append(unit_warning)

    reasons_warning = check_blocked_reasons_populated(paths)
    if reasons_warning:
        warnings.append(reasons_warning)

    if is_propagation:
        mag_warning = check_magnitude_plausibility(result)
        if mag_warning:
            warnings.append(mag_warning)

    if warnings:
        msg = " | ".join(warnings)
        sys.stderr.write(msg + "\n")
        sys.exit(2)

    # All checks pass — silent exit
    sys.exit(0)


if __name__ == "__main__":
    main()
