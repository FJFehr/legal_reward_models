# Debug Task

Goal: identify, explain, and fix a bug with minimal and reliable changes.

All working rules from `AGENTS.md` apply.

## Process

### 1. Reproduce

- State the observed behaviour.
- Identify how to reproduce it (inputs, steps, environment).
- If reproduction is not possible, say so explicitly and work with available evidence.

### 2. Isolate

- Narrow the problem to a specific component, function, or code path.
- Identify where expected vs actual behaviour diverges.

### 3. Explain

- Describe the root cause in concrete terms.
- Point to the exact code responsible.
- Explain *why* the bug happens, not just *where*.

### 4. Fix

- Make the smallest change that resolves the root cause.
- Do not introduce refactors unless necessary to fix the issue.

### 5. Add regression test

- The test should fail before the fix and pass after.
- Keep the test minimal and focused on the specific issue.

### 6. Verify

Follow the verification steps in `AGENTS.md`.

## Additional response fields

Beyond the base fields in `AGENTS.md`, include:

- **Reproduction**: steps or conditions that trigger the bug
- **Root cause**: why the bug occurs, with code references
- **Fix**: what was changed and why it resolves the issue
- **Test**: the added regression test and what it verifies
