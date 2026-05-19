# Development Workflow

## Daily Logs

Development logs live in:

- `dev_logs/`

Use one file per day:

```text
dev_logs/YYYY-MM-DD.md
```

Each log should include:

- Completed work
- Decisions made
- Verification commands run
- Known issues
- Next TODOs

## Update Rule

At the start of each development session:

1. Open today's log file.
2. Add a short session goal.
3. Record completed work as changes are made.
4. End with TODOs and verification status.

## Change Size

Keep each development step small:

- Structure first.
- Then imports and paths.
- Then behavior-preserving refactor.
- Then recommendation feature.
- Then README and GitHub cleanup.

## When to Pause

Pause for user confirmation when:

- A design choice changes the project scope.
- A change would remove or rewrite large parts of the original code.
- A dependency or tool is added beyond the current stack.
- The data format does not match assumptions.

