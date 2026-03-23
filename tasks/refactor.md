# Refactor Task

Goal: improve the codebase structure, clarity, or maintainability without changing external behaviour.

## Mindset

- Preserve behaviour exactly unless a change is explicitly requested.
- Prefer small, safe improvements over large rewrites.
- Make the code easier to understand, test, or maintain.
- Do not use refactoring as an excuse to add features.

## Process

### 1. Identify the refactor target

- State what is hard to read, hard to maintain, duplicated, or brittle.
- Explain why the refactor is worth doing.
- Confirm that the intended behaviour should stay the same.
- State explicitly what externally observable behaviour, APIs, filenames, and interfaces must remain unchanged.

### 2. Plan the change

- Identify the smallest useful scope for the refactor.
- Prefer local edits over broad restructuring.
- Reuse existing patterns where possible.
- Call out any risks to behaviour or compatibility.

### 3. Refactor

- Keep the external interface stable unless the task says otherwise.
- Make the code simpler, clearer, or more consistent.
- Remove duplication only when it genuinely improves the design.
- Avoid changing logic unless needed to preserve existing behaviour.

### 4. Add or adjust tests

- Update tests if the refactor changes internal structure in a way that needs coverage.
- Add regression tests only if the refactor exposed a fragile area.
- Tests should confirm behaviour has not changed.

### 5. Verify

- Run relevant tests.
- Check for behavioural regressions.
- Confirm the refactor is functionally equivalent to the original intent.

If full verification is not possible, use alternative evidence (manual review, static analysis, reasoning about code paths). State exactly what was checked, what remains uncertain, and why.

## Constraints

- No feature creep.
- No behaviour changes unless explicitly requested.
- No broad rewrites without clear justification.
- No unrelated cleanup.
- Explicitly list anything out of scope to prevent the refactor from quietly widening.

## Output format

Provide a structured response:

### Summary
What was refactored and why.

### Before / after
What improved in the code.

### Behaviour
Why the change is behaviour-preserving.

### Tests
What tests were run or updated.

### Verification
What was checked.

### Risks / unknowns
Anything not fully verified or worth reviewing.
