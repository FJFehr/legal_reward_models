# Debug Task

Goal: identify, explain, and fix a bug with minimal and reliable changes.

## Mindset

- Do not guess.
- Do not jump to a fix.
- First reproduce, then isolate, then explain, then fix.

## Process

### 1. Reproduce

- Clearly state the observed behaviour.
- Identify how to reproduce it (inputs, steps, environment).
- If reproduction is not possible, say so explicitly and work with the available evidence.

### 2. Isolate

- Narrow the problem to a specific component, function, or code path.
- Avoid scanning the whole codebase without direction.
- Identify where expected vs actual behaviour diverges.

### 3. Explain

- Describe the root cause in concrete terms.
- Point to the exact code responsible.
- Explain *why* the bug happens, not just *where*.

A good explanation should make the fix feel obvious.

### 4. Fix

- Make the smallest change that resolves the root cause.
- Do not introduce refactors unless necessary to fix the issue.
- Avoid changing unrelated behaviour.

### 5. Add test

- Add a regression test following the repository's existing test layout that reproduces the bug.
- The test should fail before the fix and pass after the fix.
- Keep the test minimal and focused on the specific issue.
- Do not over-generalise the test.

### 6. Verify

- Confirm the new test passes.
- Run existing tests if available.
- Check for obvious regressions.

If full verification is not possible, state what was checked and what remains uncertain.

## Constraints

- No speculative fixes.
- No broad rewrites.
- No silent assumptions.
- One bug, one minimal fix. Do not bundle unrelated changes unless the task explicitly requires it.

## Output format

Provide a structured response:

### Summary
Short description of the issue and fix.

### Reproduction
Steps or conditions that trigger the bug.

### Evidence
Error output, tracebacks, failing inputs, or reproduction notes that support the diagnosis.

### Root cause
Clear explanation of why the bug occurs, with code references.

### Fix
What was changed and why it resolves the issue.

### Test
Description of the added test and what it verifies.

### Verification
What was tested or checked.

### Risks / unknowns
Anything that could not be fully verified or might break.
