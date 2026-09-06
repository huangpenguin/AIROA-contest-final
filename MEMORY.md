# PARC2026_final Project Memory

> Purpose: durable cross-session context for Codex and human handoff. Read this file before major work. This is an index and rulebook, not a substitute for current machine evidence.
> Last memory refresh: 2026-09-06 (Asia/Tokyo), memory commit pending at authoring time.

## 1. Canonical session entrypoint

Read in this order:

1. `AGENTS.md`
2. `MEMORY.md`
3. `docs/AXIS_PI05_PLAN_INDEX.md`
4. `docs/AXIS_PI05_PROJECT_HANDOFF_20260831.md`
5. `data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json`
6. the one active plan referenced by machine JSON `next_plan`, if non-null
7. the latest audit referenced by the plan index
8. the active run's summary/cursor/PID/heartbeat, if any

Source-of-truth priority for current state:

```text
live process + immutable artifacts + current git tree
  > machine JSON current pointers
  > plan index
  > Markdown handoff
  > phase reports
  > prior conversation summaries
```

If these disagree, stop mutation, audit the discrepancy and repair the handoff in a new evidence namespace.

## 2. Current snapshot at this memory refresh

- Repository: `/home/huang/code/PARC2026_final`
- Branch: `codex/axis-pi05-audit-remediation`; no upstream recorded.
- EF0 anchor input HEAD: `38635fbf08e4e8df8c4877ddcb6b2b2ec8b9b118`; final lifecycle state is captured by the clean closeout commit.
- Latest audited execution: `b1195dd59830f7c44d831bd0007e61229446d9b5` (EF0–EF4 execution/evidence commit; independently re-audited `remediation_required`).
- Nested OpenPI: `third_party/axis_openpi` at `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`, clean at review.
- `LIBERO-plus` has pre-existing dirty files. They belong to the user/project and must not be reset, cleaned, overwritten or silently committed.
- Current state at this refresh: `track3_ef0_ef4_audited_remediation_required`.
- Active plan: `docs/AXIS_PI05_TRACK3_EF_FAILURE_PROVENANCE_AND_LIFECYCLE_REMEDIATION_PLAN.md`; only EG0–EG4 evidence-governance/code/no-physics fixture work is authorized.
- EF0–EF3 technical contracts and the corrected EF4 checkout closure are valid, but the first failed EF4 raw validation/test/partial evidence was overwritten and is currently unavailable. The same-commit `approved` audit is superseded by the independent `remediation_required` audit. No new physics, GPU, generation, replay, training, candidate, upload or submission is authorized.
- Z0–Z14 execution evidence is in `data/reports/axis_libero_30k/track3_y3_launch_phase_ledger_remediation/`; its audit is `docs/reviews/axis_pi05/TRACK3_Y3_LAUNCH_Z0_Z14_AUDIT_20260906.md`. Historical parent remains infrastructure while physical result is derived nominal family failure. Z6–Z13 remain `not_evaluated`; before any AA execution, recheck live supervisor/PID/cursor, dirty worktree and GPU boundaries. Never start a duplicate run.

The snapshot above becomes historical as soon as execution advances. Always refresh from Git, machine JSON and the live cursor rather than treating this paragraph as a lock.

## 11. Current AA0–AA14 execution snapshot (2026-09-06)

- Evidence namespace: `data/reports/axis_libero_30k/track3_z5_derived_release_and_conservative_continuation/`; AA0–AA4 passed, including old-Z `36/36` artifact closure, derived classification, real worker/finalizer regression, and AA4 clean identity freeze.
- AA5 ran exactly once: task10/demo0, seed3121, `object_frame_conservative`, CPU/no-renderer. It returned a normal process result but failed the acquisition barrier and is classified `family_failure`; its trajectory is finite `76×7` actions / `76×79` states and is not training-eligible.
- AA14 is `hard_stop`; AA6–AA13 are `not_evaluated`; counts remain `exact/pending/accepted=0/0/0`. No GPU3, generation, replay, training, ZIP, Teleop, upload or submission occurred. Next authority: `new_reviewed_plan_required`.
- Frozen execution identity: root `cf27e7b1fe9276eaf0d4285128d56586e049d28f`, tool SHA `3484ffe39d5a80277575ca006f3ea0d8f067b510779584955b833818e917cb79`, nested OpenPI `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`, CUDA visibility empty, CPU/no-renderer.

## 12. Tonight ZIP audit correction (2026-09-06)

- AA5 evidence is valid, but the AA executor implemented only AA0–AA5 and AA14. Its closeout incorrectly made all AA6–AA13 depend on support success, contrary to the plan's family-isolation rule; containment and articulation remain untested, not failed.
- Current accepted reverse data and new T3 ZIP counts are zero. The existing Aug-31 R6 FAST ZIP remains mechanically usable and hashes to `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`, but user-reported Track3 score for that model is zero.
- The TZ plan used a 23:00 JST deadline. It delivered the clearly labeled fallback after the bounded cross-family matrix found no successful family; the fallback is not Track3 improvement.

## 13. Tonight TZ0–TZ9 execution result (2026-09-06)

- Evidence namespace: `data/reports/axis_libero_30k/track3_tonight_zip_and_crossfamily_recovery/`; final state is `fallback_delivered`/hard stop pending review.
- TZ0 passed full fallback ZIP revalidation: 3401 entries, CRC passed, FAST-only layout, C5 source-success `3/3`; source and materialized ZIP are both `9383571677` bytes with SHA `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`.
- Execution claimed TZ1 passed and froze 7 identities. TZ2 consumed 7 child attempts: support 2 diagnostic family failures plus 1 preserved executor infrastructure failure, containment 2 diagnostic family failures, articulation 2 diagnostic family failures. Audit did not approve compliant TZ1/TZ2 completion because the complete scheduler/identity contract was absent and the shared infrastructure stop was bypassed across commits. No family succeeded.
- TZ3–TZ8 are `not_evaluated` because no family succeeded: GPU3 0, generation 0, replay 0, accepted reverse `0/0/0`, training false, candidate false. GPU0/official GPU/Teleop/dense full-FT/upload/submission were not used. `LIBERO-plus` pre-existing dirty was preserved.
- Final validation after the execution evidence was written: targeted `50 passed`; project `293 passed, 7 skipped, 23 warnings`; Tonight namespace strict JSON `23/23` (machine handoff included: `24/24`), LF phase CSV, py_compile/diff-check, no partial residue, and GPU0–3 empty.

## 14. Tonight TZ0–TZ9 audit result (2026-09-06)

- Verdict: `remediation_required`. The fallback is independently approved as a mechanical R6 fallback: `9383571677` bytes, SHA `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`, CRC/3401 entries/path safety/FAST-only passed, C5 source-success `3/3`. It is not Track3 improvement.
- Seven CPU child attempts exist. The first is preserved `infrastructure_failure`; the other six are valid diagnostic `family_failure` because they fail required stages with source still true and goal false. They are not accepted data and do not prove a compliant TZ2 lifecycle.
- TZ1's claimed complete scheduler was not implemented: TZ4–TZ8 are stubs and singleton/periodic heartbeat/deadline/full identity gates are absent. After the first shared infrastructure failure, commit `343c5a5` resumed the remaining identities and the aggregate cleared the failure, contrary to the frozen global-stop rule.
- Frozen seeds were recorded but never applied; three same-family pairs produced byte-identical traces. Runtime clip/contact gates are hardcoded. All six NPZ traces hash exact but contain object arrays that fail `allow_pickle=False`; TZ9 lacks the required complete artifact manifest/outer hash and post-final-artifact validation.
- ER0–ER5 remediation is complete and hard-stopped pending review. Old TZ evidence and seven identities are immutable and were not rerun. GPU0–3, official GPU, Teleop, generation/replay, accepted data, training/norm, new candidate ZIP, upload and submission remain unauthorized; next authority is `new_reviewed_plan_required`.

## 15. EF0–EF4 independent audit correction (2026-09-06)

- Audit input root is `b1195dd59830f7c44d831bd0007e61229446d9b5`; EF0 anchor is `38635fbf08e4e8df8c4877ddcb6b2b2ec8b9b118`; nested OpenPI is `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`.
- ER anchor `29/29`, EF checkout `22/22`, producer manifest `13/13`, outer/closure hashes, strict JSON, typed NPZ, LF CSV and fallback full ZIP all independently pass. The fallback remains `fallback_no_track3_improvement` with `9383571677` bytes and SHA `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`.
- Verdict is `remediation_required`: `ef4_first_attempt_failure.json` explicitly admits that the historical report was replaced by the corrected attempt. The original failed validation, full test stdout/stderr and partial inventory are not present in the repository, `/tmp`, or unreachable Git blobs and must remain labeled `unavailable`, never reconstructed as raw.
- The prior `approved` report was committed together with the EF tool/evidence and incorrectly claimed the failure was not overwritten. It remains historical but is superseded by `docs/reviews/axis_pi05/TRACK3_EF0_EF4_INDEPENDENT_AUDIT_20260906.md`.
- The only active authority is EG0–EG4: attempt-specific immutable finalizer evidence, clean-source/no-physics fixtures, lifecycle pointer consistency and hard stop. It does not authorize physics, any GPU, generation/replay, accepted data, training, ZIP creation, upload or submission.

## 3. Stable competition/model facts

- Base initialization is `axisrobotics/Pi0.5-Axis`. It is LIBERO-compatible in action representation but is not itself LIBERO post-trained; its original training-domain normalization must not be reused for our mixture.
- Valid R6 candidate: `r6_dual_l2_fresh500_r1`, topology `L2_action_lora_heads`, 500 optimizer steps. It is the continuation initialization for the active Track 3 line; do not return to Axis init unless a future audited plan explicitly authorizes it.
- User-reported official scores: Track 1 `0.47`, Track 2 `0.33`, Track 3 `0`. Treat these as user-reported official outcomes unless submission IDs/logs are later persisted; do not invent episode counts or hidden-task details.
- The Track 1/2 scores came from the R6 folded/FAST submission path, not dense full fine-tuning.
- Official public disclosure: Track 3 consists of reverse operations of existing LIBERO tasks. Exact hidden task identities/count remain private. Do not use leaked per-task information; only public rules, public datasets and public task assets are allowed.
- Track 3 zero does not prove a broken action head by itself. It currently shows that positive-task R6 supervision did not yield sufficient reverse-task competence; obtain valid reverse supervision and evaluate before assigning blame to the action head or expert module.

## 4. Data and normalization facts

- Canonical Axis reproduction data: `data/raw/hf/physical-intelligence/libero`, native 10 Hz, about 33 GiB, 1693 episode parquet files, no interpolation/resampling.
- Local competition-aligned forward source: `data/raw/hf/lerobot/libero_plus`, LeRobot v3, 20 Hz, 14347 episodes, 2238036 frames, 40 tasks at the last audit. Re-read `meta/info.json` and hashes before training.
- Track 3 reverse data must be machine-accepted only: exact start-state provenance, 20 Hz, successful goal, finite actions, contact/collision/stall/timeout gates and fresh-process replay. Failed, partial, diagnostic and holdout trajectories are denylisted from training.
- Action is 7D LIBERO OSC_POSE command `[dx,dy,dz,droll,dpitch,dyaw,gripper]`, not direct metres/radians. Persisted first three action dimensions are roughly `[-0.94,0.94]`, rotations roughly `[-0.38,0.38]`, gripper `{-1,+1}`. Do not confuse observation EEF displacement scale with action scale.
- Use per-dimension `QUANTILES` generated from the effective training manifest. Training, OpenPI rollout, folded PyTorch, LeRobot processors and FAST deployment must reference the same norm artifact/hash. Do not mix QUANTILES with `MEAN_STD`, `IDENTITY` or Axis pretraining norm.
- Current Track 3 mixture design is forward/reverse `80/20`, forward stratified by task/perturbation and reverse stratified by family then source/profile. Recompute norm after the final accepted manifest is frozen.

## 5. Current Track 3 technical state

Completed and reusable:

- Public HDF5/task/BDDL mapping and three reverse operator families.
- Same-demo composed reverse starts and nine no-pickle restorable states.
- Fresh CPU restore/static gates and GPU3 child camera pipeline.
- Atomic child result protocol, stdout/stderr redirection and timeout termination.
- task10/demo0 attempt3 child: restore/truth/contact-v3/camera all passed; two views, 41 frames each, H.264 at 20 FPS.
- Parent attempt3 recorded seed/PIDs, GPU3 UUID, start/end snapshots, 459 MiB peak, 113 samples, return code 0 and no timeout.

Still not completed at this snapshot:

- Y3 nominal CPU physics canary and all subsequent Track 3 work.
- Any successful manipulation executor pilot.
- Any reverse `exact/pending/accepted` demo; current counts remain `0/0/0`.
- Track 3 training, holdout result, FAST export or ZIP.

Latest audit correction:

- Old `parent_poll_interval_max_ms=148.4566` measured synchronous `nvidia-smi` query latency, not heartbeat gap. It must be a warning metric, not a `100 ms` hard gate.
- Old O10 manifest has 9/10 checkout hashes matching; its CSV hash was captured with CRLF and Git checked out LF. Preserve it as old failed evidence. New closure must force LF and prove checkout-stable SHA.
- Do not run task10 attempt4. The active plan authorizes a derived read-only release of attempt3 after telemetry-v2 and manifest tests pass.

## 6. Resource and execution boundaries

- Local hardware: four Tesla V100 32 GiB GPUs.
- GPU0 has repeatedly contained an external YOLO process. Never touch, kill, migrate or schedule work on GPU0, even if it later appears free, unless the user explicitly changes this rule in a reviewed plan.
- GPU1+GPU2: LoRA training when authorized. GPU3: simulator/camera/generation/replay/evaluation, one simulator process at a time.
- V100 training is FP32 under current plans; do not enable BF16. Current LoRA recipe uses microbatch 1 per device, accumulation 32, effective global batch 64.
- Competition RTX PRO 6000 Blackwell allocation is conserved for later training/final checks. Do not start/download/use it without explicit current-plan authorization.
- Training/download commands must not assume GPU use improves network downloads. Data download is CPU/network/disk work.
- Long unattended work must use a single-instance persistent supervisor (`setsid` or equivalent), PID file, command/commit/tool-hash validation, atomic cursor and heartbeat. SSH disconnect is not a failure signal.
- Before every GPU run: record `nvidia-smi` process/UUID/memory snapshot, enforce allowed IDs and preserve external processes. Afterward verify release and no orphan workers.
- Never auto-upload or submit a ZIP. Packaging may be authorized conditionally; official submission remains a user action unless explicitly delegated.

## 7. Training and evaluation gates

- R6 continuation, not Axis init, is the default for Track 3 LoRA.
- Dense full fine-tuning is a separate parked research line after repeated V100 memory-lifetime OOMs. It cannot be substituted into an active LoRA plan without a new audit/authorization.
- LoRA arm failures are isolated when possible. Nonfinite loss/grad/update/params/EMA, restore failure or data contamination is a hard stop for the affected/global scope specified by plan.
- Reverse policy evaluation must use held-out start identities, fixed policy seeds and execute8 as primary. Scripted executor success is reference/data validation, not policy success.
- Retention uses fixed C20 standard-LIBERO manifest/seeds. Current continuation gate requires at least +1 reverse success over step0 and no more than 2/20 C20 success loss.
- Only a selected checkpoint passing fold/parity, norm-hash, FAST latency, coverage and ZIP CRC/layout/size gates may become a local candidate ZIP.

## 8. Audit and planning rules

Every review must perform all of the following:

1. Read-only verify root/nested commits, dirty files, immutable input/artifact hashes, raw reports, claimed gates, cursor, processes, GPU boundary and prohibited actions.
2. Distinguish model/physics failure from evaluator, renderer, wrapper, telemetry, manifest and reporting failures.
3. Use exactly one verdict: `approved`, `remediation_required`, or `blocked`.
4. Historical evidence is immutable. Fix labels or semantics only in a new review/derived namespace with links to old hashes.
5. Missing/unrun work is `not_evaluated`; never infer success or failure and never manufacture absent evaluator/ZIP/training results.
6. Archive the completed/superseded active plan; retain it in Git. Keep exactly one new active execution plan in `docs/`.
7. Synchronize `docs/AXIS_PI05_PLAN_INDEX.md`, `docs/AXIS_PI05_PROJECT_HANDOFF_20260831.md`, machine JSON and lifecycle tests.
8. Run strict duplicate-key JSON parsing, artifact/hash validation, targeted tests, full tests, py_compile/format checks and `git diff --check` on the final would-be commit tree.
9. Commit a clean root tree. Verify nested OpenPI clean and explicitly record that the pre-existing `LIBERO-plus` dirty state was untouched.
10. A hard stop ends authorization, not necessarily technical possibility. The next step requires a new reviewed plan; do not continue the tail of an archived plan.

## 9. Git/evidence hygiene

- Preserve user changes; never use destructive reset/checkout/clean to make a tree pass.
- Small code/config/JSON/CSV/Markdown/test evidence should be tracked. Large weights/videos/NPZ/rollouts may remain ignored but must have stable manifests, bytes and SHA-256 where the plan requires them.
- Generate manifests only after producers close/fsync files. Do not include a manifest in its own content hash.
- CSV evidence uses explicit LF line endings so hashes survive Git checkout.
- Run tests again after updating handoff/status tests. A test count produced before the final handoff edit is not proof for the final commit.
- Commit hashes named in plans are inputs/provenance, not permission to rewrite the referenced history.

## 10. How to hand off to a new session

The user can say:

> 在 `/home/huang/code/PARC2026_final` 接手。先完整读取根目录 `AGENTS.md`、`MEMORY.md`，再按其中顺序读取 plan index、handoff、machine JSON、当前 active plan、latest audit 和 live cursor。先只读审计是否已有执行任务，不要启动第二实例。

After a phase completes, the reviewing agent must refresh this file only when stable project-wide facts or the canonical session entrypoint change. Fast-moving phase details belong in the machine handoff, plan index and latest audit; do not let `MEMORY.md` grow into a second execution ledger.

## 15. ER0–ER5 Executor Contract remediation result (2026-09-06)

- ER0 derived the old Tonight namespace without mutation: 23/23 JSON strict; all 6 old NPZ object-array/no-pickle failures, first infrastructure failure, six diagnostic family failures, aggregate null contradiction and three identical-trace groups were preserved. Both R6 fallback/source ZIPs were read-only checked at 9,383,571,677 bytes, SHA `95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a`, CRC passed, 3401 entries, FAST-only.
- ER4 executed exactly three new identities, in support→containment→articulation order; all three were valid diagnostic `family_failure`, with no retry and no old identity rerun. All three traces are no-pickle/finite and `training_eligible=false`; controller saturation was observed and correctly prevents success.
- ER5 delivered only the explicitly labeled R6 fallback, not Track3 improvement: `accepted=0`, training false, new candidate false. Its local JSON/NPZ/CSV/hash facts and independent targeted `15` / full `308 passed, 7 skipped, 23 warnings` reruns are valid. The audit did not approve the general executor/finalizer contract: all 29 ER evidence files were ignored, resume did not recheck full identity, required physical contact was not gated, tests were not run by ER5, and final pass omitted ZIP/process/GPU/tests. Execution source commit is `04eef24161bfe62adc9a51378235f73b34e6ac51`; nested OpenPI remains `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`.

## 16. ER0–ER5 audit and EF authorization (2026-09-06)

- Verdict: `remediation_required`. Valid facts remain immutable: fallback SHA/CRC/3401/FAST-only, old TZ failure semantics, and three ER4 family failures at 155/149/180 steps with source true, goal false and typed no-pickle traces.
- ER evidence is locally complete enough to inspect but not checkout-stable: `0/29` files are tracked. The 26-entry manifest and outer hash are exact on the current machine, not anchored in any execution commit.
- The next plan is EF0–EF4 only: exact-byte Git anchoring, full parent/child/resume identity, real no-physics supervisor regressions, required-contact gate semantics and a finalizer whose pass status covers ZIP/process/GPU/tests. All consumed identities and all GPU/data/training/ZIP expansion remain prohibited.

## 17. EF0–EF4 Evidence Anchor / Finalizer result (2026-09-06)

- EF0 anchored all 29 ER evidence files byte-for-byte in commit `38635fbf08e4e8df8c4877ddcb6b2b2ec8b9b118`; commit read/hash and worktree hashes were `29/29` exact. The old ER namespace persistence distinction remains explicit: 24 Git-persisted records and 6 local NPZ records.
- EF1 froze parent-provided root/tool/OpenPI/profile/command, porcelain and digest fields plus CPU/no-renderer environment; real drift probes and legal resume passed, while trace bytes/SHA drift failed closed. EF2's no-physics fixture passed singleton lock, heartbeat sequence, TERM→KILL, no-descendant and sticky stop. EF3 separated telemetry schema completeness from physical required contact with pass/fail/not_evaluated fixtures.
- EF4's new namespace records the real fallback ZIP check, process/GPU scan, targeted/full tests, manifest/outer hash, final validation and `state=closed` heartbeat. No physics identity, GPU compute, generation, replay, accepted data, training, candidate ZIP, upload or submission occurred. Independent audit is `approved`; no active plan remains.
