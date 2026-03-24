# Legal Reward Models

A research pipeline for training and evaluating reward models on unverifiable data using RLAIF and SFT.

## Pipeline overview

1. **Data analysis** — explore and understand source legal datasets
2. **Preference pair construction** — convert data into (chosen, rejected) pairs for reward modelling
3. **Training** — fine-tune models via SFT and RLAIF
4. **Evaluation** — assess reward model quality on unverifiable legal data

## Setup

```bash
uv sync
```

## Running tests

```bash
uv run pytest tests/
```

## Repository layout

| Path | Role |
|------|------|
| `scripts/` | Runnable scripts (e.g. vLLM inference) |
| `notebooks/` | Jupyter notebooks for data exploration and EDA |
| `tests/` | Pytest test suite |
| `data/` | Local data directory (git-ignored, kept via `.gitkeep`) |
| `AGENTS.md` | Operating contract for coding agents |
| `tasks/` | Task briefs by type: feature, refactor, debug |
| `orchestrator_prompt.md` | Prompt template for dispatching agent tasks |

### Read order

1. `AGENTS.md` — rules and priorities
2. The relevant `tasks/*.md` brief
3. `orchestrator_prompt.md` — how tasks are dispatched
4. This file — setup and project overview

## Working in this repo

See `AGENTS.md` for coding conventions and task workflow.
