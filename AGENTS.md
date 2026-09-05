# PARC2026_final Agent Instructions

These instructions apply to the whole repository.

## Start every new session here

Before planning, editing, training, evaluating, or answering project-status questions:

1. Read `MEMORY.md` completely.
2. Read `docs/AXIS_PI05_PLAN_INDEX.md`.
3. Read the top-level current fields and latest relevant section of `docs/AXIS_PI05_PROJECT_HANDOFF_20260831.md`.
4. Parse `data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json` with duplicate-key rejection.
5. Read the single active plan named by `next_plan` and the latest audit named by the plan index.
6. Run read-only `git status`, root/nested HEAD checks, process/GPU checks, and inspect the active plan's cursor/PID/heartbeat before deciding anything.

Never reconstruct current state from chat history alone. The repository handoff and immutable evidence are authoritative.

## Do not collide with a running task

- If the active plan has a live supervisor/PID or another agent has a dirty worktree, do not launch a second run, rewrite its cursor, archive its plan, or commit over its changes.
- Observe and report unless explicitly asked to take over. Preserve user/other-agent modifications.
- A disconnected SSH/Codex session does not imply the run stopped; verify PID, command, heartbeat, cursor, GPU and atomic artifacts.

## Mandatory audit lifecycle

For every user-requested phase review:

1. Independently verify commit, hashes, raw artifacts, gates, tests, process/GPU boundary and all claimed non-actions.
2. Return exactly one audit class: `approved`, `remediation_required`, or `blocked`.
3. Preserve historical reports and failure labels. Corrections go into a new audit/derived namespace; never rewrite old evidence to make it pass.
4. Archive the completed/superseded active plan under `docs/archive/axis_pi05/`.
5. Create exactly one new active plan under `docs/` if more work is authorized.
6. Update the plan index, Markdown handoff, machine JSON handoff and lifecycle tests together.
7. Re-run strict JSON, targeted tests, full tests and `git diff --check` on the final tree, then make a clean commit.
8. Mark unexecuted work `not_evaluated`; never convert missing work to false, pass, success, zero-risk, or inferred evidence.

Detailed rules and current facts live in `MEMORY.md`; do not duplicate or weaken them here.
