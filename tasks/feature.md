# Feature Task

Goal: add a new capability with minimal, well-scoped changes.

## Mindset

- Understand the requested outcome before touching code.
- Prefer the smallest implementation that satisfies the requirement.
- Keep the change aligned with the existing codebase.
- Do not add extra features just because they seem useful.

## Process

### 1. Clarify the feature

- Restate the requested behaviour in concrete terms.
- Define the acceptance criteria for the feature.
- Identify the user-facing outcome and the internal impact.
- State what existing behaviour must remain unchanged.
- Separate required behaviour from non-goals and nice-to-have ideas.
- If requirements are ambiguous, call out the uncertainty before proceeding.

### 2. Plan the change

- Identify the files, modules, or components likely to be affected.
- Keep the implementation path simple.
- Prefer extending existing patterns over introducing new ones.
- Distinguish required user-visible behaviour from internal implementation choices.
- Note any dependencies, constraints, or trade-offs.

### 3. Implement

- Make the smallest change that delivers the feature.
- Keep the code readable and consistent with the repository.
- Avoid unrelated refactors.
- Avoid broad structural changes unless the feature genuinely requires them.

### 4. Add tests

- Add or update tests in the repository's existing test location for the new behaviour.
- Cover the main success path, important edge cases, and relevant failure modes.
- Keep tests focused on the feature, not implementation details.
- The tests should clearly show the feature works as intended.

### 5. Verify

- Run the relevant tests.
- Check that existing behaviour still works.
- Confirm the feature behaves as described.
- If something cannot be verified, say so clearly.

## Constraints

- No speculative additions.
- No unnecessary abstraction.
- No unrelated clean-up.
- No hidden changes to existing behaviour unless explicitly required.

## Output format

Provide a structured response:

### Summary
What feature was added.

### Behaviour
What the feature does now.

### Implementation
What changed and where.

### Tests
What tests were added or updated.

### Verification
What was checked.

### Risks / unknowns
Anything uncertain, incomplete, or worth follow-up.
