#!/usr/bin/env python3
"""
Arithmetic verifier for economics analysis.

Reads a Stop hook JSON payload from stdin, extracts the last assistant message,
finds arithmetic expressions (A * B = C, X% of Y = Z, etc.), and verifies them.

Exit codes:
  0 — No errors found (or non-applicable content)
  Non-zero — Should not happen; errors reported via JSON stdout

Output (only when errors found):
  {"ok": false, "reason": "Arithmetic error: ..."}

Stdlib-only. No external dependencies.
"""

import json
import re
import sys


# Tolerance for floating point comparisons (2%)
TOLERANCE = 0.02


def parse_number(s: str) -> float | None:
    """Parse a number string, handling commas, $, %, and common prefixes."""
    s = s.strip()
    # Remove currency symbols and commas
    s = re.sub(r'[$€£¥]', '', s)
    s = s.replace(',', '')
    # Remove trailing %
    s = s.rstrip('%')
    # Handle multiplier suffixes
    multipliers = {
        'k': 1e3, 'K': 1e3,
        'M': 1e6, 'm': 1e6,
        'B': 1e9, 'b': 1e9,
        'T': 1e12, 't': 1e12,
    }
    for suffix, mult in multipliers.items():
        if s.endswith(suffix) and len(s) > 1:
            try:
                return float(s[:-1]) * mult
            except ValueError:
                continue
    try:
        return float(s)
    except ValueError:
        return None


def close_enough(expected: float, actual: float) -> bool:
    """Check if two values are within TOLERANCE of each other."""
    if expected == 0 and actual == 0:
        return True
    if expected == 0:
        return abs(actual) < 0.01
    return abs((actual - expected) / expected) <= TOLERANCE


def find_and_check_arithmetic(text: str) -> list[str]:
    """Find explicit arithmetic expressions and verify them.

    Patterns matched:
    - A × B = C  /  A * B = C  /  A x B = C
    - A + B = C
    - A - B = C
    - A / B = C  /  A ÷ B = C
    - X% of Y = Z  /  X% × Y = Z
    """
    errors = []

    # Number pattern: optional negative, digits with optional commas, optional decimal,
    # optional suffix (K/M/B/T), optional currency prefix
    num = r'[$€£¥]?\-?\d[\d,]*\.?\d*[KkMmBbTt]?%?'

    # Pattern: A op B = C
    # Operators: ×, *, x (multiply), +, -, /, ÷
    pattern = rf'({num})\s*([×*x+\-/÷])\s*({num})\s*=\s*({num})'

    for match in re.finditer(pattern, text):
        a_str, op, b_str, c_str = match.groups()

        # Skip if 'x' operator is likely part of a word (e.g., "2x leverage")
        if op == 'x' and not re.match(r'\s', match.group()[match.group().index('x')-1:match.group().index('x')]):
            # Be conservative — only match ' x ' with spaces
            context = text[max(0, match.start()-1):match.end()+1]
            if ' x ' not in context and ' × ' not in context:
                continue

        a = parse_number(a_str)
        b = parse_number(b_str)
        c = parse_number(c_str)

        if a is None or b is None or c is None:
            continue

        # Handle percentage: if a_str ends with %, divide by 100
        if a_str.rstrip().endswith('%'):
            a = a / 100
        if b_str.rstrip().endswith('%'):
            b = b / 100

        if op in ('×', '*', 'x'):
            expected = a * b
        elif op == '+':
            expected = a + b
        elif op == '-':
            expected = a - b
        elif op in ('/', '÷'):
            if b == 0:
                errors.append(f"Division by zero: {match.group()}")
                continue
            expected = a / b
        else:
            continue

        if not close_enough(expected, c):
            errors.append(
                f"Arithmetic error: {a_str} {op} {b_str} should be "
                f"{expected:,.4g}, not {c_str} (found: {match.group()})"
            )

    # Pattern: X% of Y = Z
    pct_of = rf'({num})%\s+of\s+({num})\s*=\s*({num})'
    for match in re.finditer(pct_of, text):
        pct_str, base_str, result_str = match.groups()
        pct = parse_number(pct_str)
        base = parse_number(base_str)
        result = parse_number(result_str)

        if pct is None or base is None or result is None:
            continue

        expected = (pct / 100) * base
        if not close_enough(expected, result):
            errors.append(
                f"Arithmetic error: {pct_str}% of {base_str} should be "
                f"{expected:,.4g}, not {result_str} (found: {match.group()})"
            )

    return errors


def main():
    # Read JSON from stdin
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        # No valid input — pass through silently
        sys.exit(0)

    # Loop guard: if we're already in a hook, don't recurse
    if data.get("stop_hook_active"):
        sys.exit(0)

    # Extract the assistant's last message
    message = data.get("last_assistant_message", "")
    if not message:
        sys.exit(0)

    # Run arithmetic checks
    errors = find_and_check_arithmetic(message)

    if errors:
        result = {
            "ok": False,
            "reason": "; ".join(errors)
        }
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
    # If no errors, output nothing (silent pass)


if __name__ == "__main__":
    main()
