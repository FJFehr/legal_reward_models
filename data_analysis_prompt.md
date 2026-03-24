# Orchestrator Prompt

Instruction precedence: AGENTS.md > this prompt > task brief > README.md

## Required fields

**Task type:** [feature | debug | refactor | docs]
**Mode:** [propose-only | implement]
**Objective:** One sentence stating the concrete outcome.
**Acceptance criteria:** What done looks like. Measurable if possible.
**Constraints:** Things that must not change or must not be added.
**Known unknowns:** Open questions or unclear inputs.
**Relevant files:** List of files the agent should read.
**Out-of-scope files:** Files that must not be modified.
**Verification:** How to confirm the task is complete.

## Execution mode

- `propose-only` — plan and explain only. Do not edit files or run mutating commands.
- `implement` — plan, then make the changes, then verify.

## Ambiguity rule

Do not ask for clarification on every uncertainty. Ask only when the ambiguity could materially affect correctness and you cannot resolve it conservatively. Otherwise: state the assumption, proceed, and flag it in the response.

## Output format

Follow the output format in the relevant task brief (`tasks/<type>.md`).

---

## TASK
Instruction precedence: AGENTS.md > this prompt > task brief > README.md

**Task type:** feature
**Mode:** implement
**Objective:** Add a data folder and a EDA exploratory data anlaysis notebook. This script will download a huggingface dataset for example https://huggingface.co/datasets/LEXam-Benchmark/LEXam or https://huggingface.co/datasets/isaacus/legal-rag-bench and provide a way to view the data and get data statistics such as how many items in train, test, val, plots of the categories or metadata that may be relevent. Print out an example for manual inspection etc.
**Acceptance criteria:**
**Known unknowns:** I am not sure yet how to run and visualise notebooks in vscode
**Relevant files:**
**Out-of-scope files:**
**Verification:**
