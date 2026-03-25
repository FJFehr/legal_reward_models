# STYLE.md

This file defines the default coding style for this repository.

If the existing codebase consistently contradicts this style, follow the codebase and document the deviation.

---

## 1. Core flavour

- Prefer pragmatic, task-driven engineering over abstract design.
- Optimise for readability, correctness, and ease of modification.
- Write code for the next person (often yourself), not for aesthetic purity.

---

## 2. Structure and modularity

- Organise code by responsibility (core logic, orchestration, tests, docs).
- Keep boundaries clear between reusable logic, entrypoints, and experiments.
- Prefer small, well-defined modules with clear entrypoints.
- Keep helpers local when used in one place; extract shared logic only when reuse is clear and stable.
- Prefer simple composition over deep abstraction hierarchies.

Avoid:
- blurred module responsibilities or "misc" catch-all folders
- premature abstraction or deep indirect call chains without clear benefit
- mixing experimental code into stable modules
- committing runtime or generated artefacts

---

## 3. Naming

- `snake_case` for files, functions, and variables.
- `PascalCase` for classes and types.
- Prefer descriptive names over short or clever ones.
- Test names should describe behaviour (`test_<what>`).

---

## 4. Syntax and formatting

- Follow standard Python formatting: 4-space indentation, consistent line wrapping.
- Use formatting tools (e.g. Black, isort) where available.

### Imports
- Group imports: standard library, third-party, local modules.
- No wildcard imports.

### Comments
- Explain intent, non-obvious logic, and important trade-offs.
- Keep comments concise. Do not restate obvious code.

### Docstrings
- Short, practical docstrings for non-trivial functions and classes.
- Prefer explaining inputs, outputs, and behaviour over prose.

---

## 5. Error handling

- Fail fast on invalid states using explicit exceptions.
- Use simple guards and checks where appropriate.

Avoid:
- silent failures
- overly complex retry logic unless clearly necessary
- using assertions for user-facing validation

---

## 6. Tests

- Verify meaningful behaviour: correctness, regression, equivalence.
- Use clear assertions and minimal setup.
- Keep tests readable and focused on what is being verified.

Avoid:
- heavy or implicit setup at import time
- reliance on global state or path hacks
- tests that obscure what is being verified

---

## 7. Documentation

- Documentation should be practical: how to run, how to use, what the code does.
- Keep tone clear and direct.

Avoid:
- stale or incomplete sections
- overly verbose explanations without practical value
