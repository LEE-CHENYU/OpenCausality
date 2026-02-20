---
name: econ-review
description: >
  Run a deep 7-point economics verification on recent analysis.
  Checks sign conventions, mechanism attribution, horizon consistency,
  causal direction, unit analysis, stock/flow classification, and
  real/nominal specification.
user-invocable: true
---

# /econ-review — Economics Verification

Run the `econ-verifier` agent to perform a deep 7-point economics review.

## Instructions

When this skill is invoked:

1. **Identify the target text**: If the user provided specific text or a file path as an argument, use that. Otherwise, review the most recent substantive economics analysis in the conversation.

2. **Launch the econ-verifier agent**: Use the Task tool to spawn the `econ-verifier` agent with the target text. The agent will run all 7 verification checks.

3. **Present the results**: Show the verification report to the user. Highlight any FAIL or WARNING items prominently.

## Usage Examples

- `/econ-review` — Review the most recent economics analysis in conversation
- `/econ-review outputs/analysis.md` — Review a specific file
- `/econ-review "The elasticity of oil demand is -0.3..."` — Review specific text
