# Feature Task

Goal: add a new capability with minimal, well-scoped changes.

All working rules from `AGENTS.md` apply.

## Process

### 1. Clarify the feature

- Restate the requested behaviour in concrete terms.
- Define acceptance criteria.
- Identify user-facing outcome and internal impact.
- Separate required behaviour from non-goals.
- If requirements are ambiguous, call out the uncertainty before proceeding.

### 2. Plan the change

- Identify affected files, modules, or components.
- Prefer extending existing patterns over introducing new ones.
- Note any dependencies, constraints, or trade-offs.

### 3. Implement

- Keep the code readable and consistent with the repository.
- Avoid broad structural changes unless the feature genuinely requires them.

### 4. Add tests

- Cover the main success path, important edge cases, and relevant failure modes.
- Keep tests focused on the feature, not implementation details.

### 5. Verify

Follow the verification steps in `AGENTS.md`.

## Additional response fields

Beyond the base fields in `AGENTS.md`, include:

- **Behaviour**: what the feature does now
- **Implementation**: what changed and where
- **Tests**: what tests were added or updated
