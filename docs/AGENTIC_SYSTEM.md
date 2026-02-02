# Agentic Causal DAG System

This repository now includes scaffolding for a multi-agent causal DAG workflow.
The goal is to turn a DAG + identification constraints into data acquisition,
model specs, estimation, diagnostics, and system-level evaluation.

## Inputs

- DAG schema: config/agentic/dag_schema.yaml
- Design registry: config/agentic/design_registry.yaml
- Example DAG: config/agentic/dags/example_kz_welfare.yaml

## Agent roles and outputs

DataScout
- Inputs: DAG NodeSpecs, shared/study data pipelines
- Outputs: DataCards in outputs/agentic/cards/data

ModelSmith
- Inputs: DAG EdgeSpecs + DataCards + design registry
- Outputs: ModelSpecs in outputs/agentic/cards/models

Estimator
- Inputs: ModelSpecs + data pipelines
- Outputs: ResultCards/EdgeCards in outputs/agentic/cards/results

Judge
- Inputs: ResultCards + ModelSpecs + DAG
- Outputs: System report + edge scores in outputs/agentic/cards/judge

## Loop scripts

Loop scripts are in scripts/agent_loops. Each agent has a dedicated loop script
that wraps `codex exec` and writes logs, PID files, and stopfiles.

Examples:
- bash scripts/agent_loops/datascout_loop.sh
- bash scripts/agent_loops/modelsmith_loop.sh
- bash scripts/agent_loops/estimator_loop.sh
- bash scripts/agent_loops/judge_loop.sh

Single-round execution (one pass through all agents):
- bash scripts/agent_loops/run_agent_round.sh

Stop a loop by creating the stopfile in outputs/agentic/<agent>/<agent>_loop.stop.

## Backward compatibility

The agentic flow is designed to reuse existing data pipelines and model code:
- shared/data and studies/*/src/data provide connectors and transformations.
- shared/model and studies/*/src/model provide estimation and diagnostics tools.

No existing data is moved or renamed; the new system layers on top of the current
structure and uses outputs/agentic for generated artifacts.
