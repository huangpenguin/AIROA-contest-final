# PARC2026_final Agent Instructions

These instructions apply to the whole repository.

## 2026-09-30 stage transition

For new competition design work, start with `docs/COMPETITION_PHASE_TRANSITION_20260930.md`. The PARC 2026 execution instructions below are historical and apply when explicitly continuing that experiment. Do not infer authorization to resume its training, evaluation, packaging, or submission from the new design stage. Preserve the local data, generated artifacts, and the separately modified `third_party/axis_openpi` checkout. Check live processes before changing any of their files.

## Historical PARC 2026 session entrypoint

When explicitly continuing PARC 2026 execution, before planning, editing, training, evaluating, or answering project-status questions:

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

## Default lightweight iteration lifecycle (user override, 2026-09-09)

Automatic independent audit is disabled. For ordinary Track3 collection and rapid iteration:

1. Check only whether the previous requested outcome exists and whether a live process or another dirty writer makes continuation unsafe.
2. Report the previous step as complete or incomplete, then land the next concrete plan requested by the user.
3. Do not automatically create an audit plan, return an audit class, archive plans, synchronize every handoff, run the full test suite, or require a clean commit.
4. A failed capture, blocked Save, or replay failure affects that episode only. Preserve the raw record, Reset or move to the next task, and continue the authorized queue.
5. Keep historical evidence immutable and mark work that was not run as `not_evaluated`; never invent missing results.
6. For data-only collection, use lightweight schema/finite/frame/hash checks. Run focused tests when recorder or task code changes. Full lifecycle tests are reserved for an explicitly requested formal audit or a release/ZIP/submission boundary.

Use the former strict audit lifecycle only when the user explicitly asks for a **formal audit**, or before an irreversible release, upload, or competition submission. This section overrides the older default-audit wording in `MEMORY.md` for routine collection work.
