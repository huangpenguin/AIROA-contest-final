# PARC2026_final Project Memory

> Purpose: durable cross-session context for Codex and human handoff. Read this file before major work. This is an index and rulebook, not a substitute for current machine evidence.
> Last memory refresh: 2026-09-07 (Asia/Tokyo), SR0–SR13 independent audit / task10 user-Teleop gold lifecycle.

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
- Branch at latest audit input: `codex/axis-pi05-sg0-sg12-support-gold-20260907`; upstream remains the same-name origin branch at `55acd39c17d82c89b4f8e809b8f84d9623c3eb81`.
- EF0 anchor input HEAD: `38635fbf08e4e8df8c4877ddcb6b2b2ec8b9b118`; its historical lifecycle is closed.
- Latest audit input / SR execution record is `64f278c7f59dc70f17770e1980522d0313ccda27`; SR physics source is `5c5c74641ba39e41709e2c51ce576e7c065060fc`. SR0–SR13 is independently audited `remediation_required`.
- Nested OpenPI: `third_party/axis_openpi` at `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`, clean at review.
- `LIBERO-plus` has pre-existing dirty files. They belong to the user/project and must not be reset, cleaned, overwritten or silently committed.
- Current state at this refresh: `track3_task10_user_teleop_ready_for_user_calibration`.
- Active plan: `docs/AXIS_PI05_TRACK3_SR_CLOSURE_AND_TASK10_USER_TELEOP_GOLD_CANARY_PLAN.md`; UT0–UT3 passed. Next executable authority is explicit user takeover (`接管 Teleop`) for UT4/UT5.
- SR0–SR13 executed and hard-stopped. Two CPU/no-renderer support identities remain diagnostic `family_failure`. UT1 derived the 18-file complete set and SR9–SR11 `not_evaluated` without rewriting attempt_0004. UT2 repaired relative-path finalizer/`not_evaluated` derivation. UT3 Teleop defaults to OPEN with explicit Open/Close and per-step gold telemetry. Do not start the simulator until the user replies `接管 Teleop`.
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

## 8. Lightweight review and planning rules (user override, 2026-09-09)

Routine Track3 data collection no longer triggers an automatic independent audit. The normal loop is:

1. Read-only check that the previous requested artifact/result exists and that no recorder, simulator, trainer, cursor, or dirty writer would collide with the next step.
2. Say only whether the previous step is complete, then provide or land the next practical plan.
3. Preserve raw history and use `not_evaluated` for work that did not run. A bad episode is excluded or re-recorded; it does not automatically stop unrelated tasks.
4. Data-only checks are schema, finite values, aligned frame counts, paths and hashes. Recorder/task code changes get focused tests. Do not run the full suite or create audit bureaucracy for every demo.
5. Do not automatically require an audit verdict, audit plan, lifecycle archive, all-handoff synchronization, or clean commit during fast collection.

The strict audit workflow is opt-in: use it only when the user explicitly requests a **formal audit**, or at a release/ZIP/upload/submission boundary where immutable provenance and broad validation are required. The safety rules for process/GPU collision and preservation of user changes remain mandatory.

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

## 18. EG0–EG4 re-audit and Acquisition Canary authority (2026-09-06)

- EG evidence commit `5d20e25d3890e937c0278ea2656236449b82898f` is byte-exact for `110/110` tracked files; the tool matches execution source `0799142b92ad3e799dd2d1b78ca964f07dd87587`. All five attempts and root attempt_0005 manifest/outer/closure links pass. Shared root attempt_0003 remains immutable failed history; attempt_0005 is the current successful pointer. Verdict: `approved`.
- The unrecoverable first EF4 validation/test/partial raw remains `unavailable`. The current-checkout pre-lifecycle strict JSON count is `96/96`; the older `94/94` count belongs to the evidence-commit boundary. No historical namespace was rewritten.
- The completed AC0–AC6 plan is archived at `docs/archive/axis_pi05/AXIS_PI05_TRACK3_CROSSFAMILY_ACQUISITION_RECOVERY_CANARY_PLAN.md`. It performed read-only trace diagnosis and code/fixture gates, then ran at most one fresh CPU/no-renderer identity per support/containment/articulation family, three total and no retry; all three were isolated normal family failures. Its independent audit plan is also archived; the active authority is now the BR remediation plan.
- Success remains only `pending_replay_review`. Old identity reruns, every GPU, official GPU, Teleop, generation/replay, accepted data, training/norm, FAST/export, candidate ZIP, upload and submission remain prohibited.
- Post-lifecycle validation: strict JSON `97/97`, targeted `19 passed`, full `339 passed, 7 skipped, 23 warnings`, py_compile/diff-check/no-partial/process/GPU boundaries passed. Nested OpenPI was clean and the pre-existing `LIBERO-plus` dirty set was preserved.

## 19. AC0–AC6 Cross-family Acquisition Canary result (2026-09-06)

- Source commit for the new executor/fixture: `afabffdb2124a870dd16bd9e2fcf3ba9df246327`. Evidence is in `data/reports/axis_libero_30k/track3_crossfamily_acquisition_recovery_canary/attempts/attempt_0001/`; `latest_attempt.json` is the only current pointer.
- AC0–AC3 passed. Old ER typed traces were re-read only at `155/149/180` steps. Historical contact pairs were absent and remain `not_evaluated`; no pair data was reconstructed. Runtime-derived contact-role, saturation, stage, source/goal and typed-trace gates were exercised, together with clean-source identity/resume/supervisor fixtures.
- AC4 used exactly three new CPU/no-renderer identities in fixed support→containment→articulation order, no retry: `ac_support_remove_20260906_0001`, `ac_containment_remove_20260906_0001`, and `ac_articulation_close_20260906_0001`. Each is an isolated normal `family_failure`; no shared infrastructure failure occurred. No success, pending replay, exact, accepted or training-eligible data was produced.
- AC3 timeout teardown left one 190-byte atomic heartbeat partial. Its bytes/SHA are preserved in `ac3_partial_residue_observation.json`; the ephemeral partial was removed before AC6. This is a provenance observation, not reconstructed raw evidence, and must not be erased or described as never having occurred.
- AC6 finalizer passed and hard-stopped: strict JSON `20/20`, targeted `7 passed`, full `346 passed, 7 skipped, 23 warnings`, manifest `28` entries, outer/pointer/closure exact, no final partial, no live Track3 workers, GPU0–3 compute empty, closed heartbeat and inactive cursor. No physics/GPU/old identity/generation/replay/training/ZIP/upload/submission beyond the three authorized CPU canaries occurred.
- At AC closeout the execution plan was archived and the read-only independent audit plan became active. That audit is now complete and archived; see section 20 for current authority.

## 20. AC0–AC6 independent audit and BR0–BR11 authority (2026-09-07)

- Verdict: `remediation_required`. AC `38/38` tracked bytes, 22 input locks, manifest `28/28`, outer/closure/pointer, current strict JSON `24/24`, 4/4 typed NPZ and LF CSV pass. The three child runs are valid diagnostic family failures at `155/149/180` steps with saturation `10/5/10`, source always true and goal always false.
- AC3 observed a 190-byte timeout heartbeat partial but still reported passed and launched AC4, contrary to the frozen shared-infrastructure sticky stop. The derived observation is honest; clearing the raw residue cannot retroactively release AC4.
- Runtime defects are independently reproduced: later-stage gates are failed instead of `not_evaluated`; `numpy.bool_ is False` prevents source-release success; contact after the acquisition barrier can count backward; attempt ID 1 can be overwritten; nested fixture heartbeats are omitted from the manifest; AC6 remains `running` in summary/CSV.
- The sole active plan is `docs/AXIS_PI05_TRACK3_ACQUISITION_GATE_AND_ATTEMPT_LIFECYCLE_REMEDIATION_PLAN.md`. BR0–BR8 may proceed continuously through code/fixture and clean-source release. Only if all pass may BR9–BR10 run three fresh CPU/no-renderer identities, one per family and no retry; BR11 then hard-stops. No GPU, replay, generation, accepted data, training, FAST/ZIP, upload or submission is authorized.
- Post-lifecycle validation: strict JSON `26/26`, targeted `14 passed`, full `353 passed, 7 skipped, 23 warnings`, py_compile/diff-check/no-partial/process/GPU boundaries passed; nested OpenPI clean and the pre-existing `LIBERO-plus` dirty set preserved.

## 21. BR0–BR11 independent audit and CR0–CR14 authority (2026-09-07)

- Verdict: `remediation_required`. BR evidence commit `474d36d7e36aa3f428c8fff3fbf38b8d5db03f05` has `88/88` exact tracked files; clean-source code is `cba0b12c578412d741b6e071c09b19caf3cbe405`. Attempts `0001–0006`, `71/71` BR JSON, pointers, typed traces, tests and current process/GPU/Git boundaries are readable and stable.
- `run_br9_to_br11()` does not re-run the clean-source gate before allocation/spawn, so dirty expected and observed identity can pass together. Attempt manifests seal too early and omit later producers/finalizer files. Attempt `0002` contains the actual 43-byte timeout partial outside its manifest; the separately written 39-byte sequence=999 observation is synthetic, not that raw partial.
- Telemetry schema unknown is incorrectly mapped to `family_failure`; normalized controller actions are divided by physical scales a second time, producing `normalized_action_abs_max=20.0`. BR's three traces are byte-identical to the corresponding AC traces and add no behavioral diversity. They remain diagnostic dirty-source family failures, never training data.
- The sole active plan is `docs/AXIS_PI05_TRACK3_CLEAN_LAUNCH_AND_DIVERSIFIED_CANARY_RECOVERY_PLAN.md`. CR0–CR8 is code/fixture/clean-source remediation. Only after those gates pass may CR9–CR10 run one primary plus at most one structurally different alternative per family, six CPU/no-renderer identities total. A normal family failure may use its frozen alternative; any shared infrastructure failure stops globally.
- CPU success may release GPU3 exact-action camera replay only and remains pending human review. With no success, prepare a human handoff but do not run unattended Teleop. GPU0–2, official GPU, generation, accepted data, training/norm, new ZIP, upload and submission remain prohibited; CR13 only prepares a machine-ready dry-run bundle.

## 22. CR0–CR14 independent audit and support-gold authority (2026-09-07)

- Verdict: `remediation_required`. CR `69/69` strict JSON, source manifest `34/34` and complete set `38/38`, derived correction manifest `2/2` and complete set `6/6`, closure/pointers, six child identities/self-hashes/typed traces/action hashes and zero process/GPU boundary pass. Six traces are byte/action-new and remain valid acquisition-before-barrier diagnostic family failures; accepted/training/new ZIP remain zero.
- Runtime contact evidence is invalid: `contact_matrix.complete` was copied to required contact and telemetry-known while unrelated contact was hardcoded false. The six traces therefore report required contact for all `966/966` steps. Do not use those fields for gold/accepted/training decisions.
- The frozen `waypoint_guarded_microstep_v1` contract was not implemented: only static target coordinates changed, with no XY/Z microstep state machine or lower cap, and every CR trace has zero rotation action. CR12's recorder argv is also not executable (`--task` instead of `--record --task-index`, missing project venv); CR13 is not a full machine-ready bundle.
- User strategy input `PARC_TRACK3_MASTER_PLAN_20260907.md` is tracked unchanged at SHA `8c2f87ebdb4c3bfccfe1a2e80294ad3d23e300f76da869f8d7db13b9e52715f5`. Its data-first/one-family/gold-demo path is adopted; its external sources and deadline claim were not independently verified by this audit.
- The sole active authority is `docs/AXIS_PI05_TRACK3_CONTACT_TRUTH_AND_SUPPORT_GOLD_DEMO_PLAN.md`: fix contact truth, real failure injection, waypoint/orientation and executable Teleop/replay handoff, then run at most two fresh support CPU/no-renderer identities. If automatic acquisition fails, wait for explicit online human control; only real success releases up to three GPU3 exact-action replays. Stop at `pending_gold_review`; no accepted promotion, training, new ZIP, upload or submission.
- Post-lifecycle validation: CR strict JSON `69/69`, audit summary and machine handoff `1/1` each, targeted `22 passed`, project suite `388 passed, 7 skipped, 23 warnings`; diff/process/GPU/nested Git boundaries passed.

## 23. SG0–SG6 independent audit and support runtime-gate authority (2026-09-07)

- Verdict: `remediation_required`. SG0–SG5 passed on attempt `0003`; SG6 attempt `0004` failed on an offset keyword TypeError before `_configure_public_env` and `ControlEnv` creation. Its 37 JSON files, 8-entry final manifest, outer/closure, cursor/heartbeat and zero-partial state pass. It is one launch attempt but zero environment/physics identities and generated no trajectory or episode.
- The `3e156dd` signature fix is valid but does not make the success path releasable. The runtime contact gate requires every sample since episode start to have demonstrator contact; it requires source=false at grasp dwell before lift; it classifies legal bowl↔main_table goal contact as unrelated; and its continuous-goal gate checks only the final frame.
- There is no current human episode review queue: no SG trajectory, camera/replay video, Teleop recording or pending gold candidate. Human action is needed only later for explicit Teleop takeover after automatic failure, or visual review after exact replay `3/3`.
- The sole active authority is `docs/AXIS_PI05_TRACK3_SUPPORT_RUNTIME_GATE_REMEDIATION_AND_GOLD_CONTINUATION_PLAN.md`. SR0–SR5 repairs contact roles, phase windows, pre-env integration and scoped stop accounting, then creates a new clean release. It allows one corrected primary physics identity and at most one progress-gated structural alternative; no same-strategy retry. A proven pre-env deterministic exception stops the child but may use one preauthorized code-only repair; identity/hash/telemetry/GPU/partial/supervisor uncertainty remains a sticky global stop.
- Real success alone may release GPU3 exact-action replay `3/3` and a front/wrist visual review card. Automatic failure may request one user-controlled Teleop candidate. Accepted promotion, training/norm, FAST/ZIP, upload and submission remain prohibited.

## 24. SR0–SR13 independent audit and user-Teleop gold authority (2026-09-07)

- Verdict: `remediation_required`. Audit input/execution record is `64f278c7f59dc70f17770e1980522d0313ccda27`; physics clean source is `5c5c74641ba39e41709e2c51ce576e7c065060fc`; nested OpenPI remains `b58aff0750f7a9ee07f3feb5df951a20ad2c76e1`.
- Both fresh support CPU/no-renderer traces are valid diagnostic `family_failure`: 96/112 steps, typed no-pickle finite states/actions, grasp-dwell reached, and the final four dwell samples have complete telemetry but no required/demonstrator contact. Lift/place are `not_evaluated`; success/pending/accepted/training remain zero.
- The SR finalizer sealed a 15-entry set before SR8/SR12/SR13 wrote three more producer files, so old closure exactness is stale. SR13 also incorrectly stores an empty `not_evaluated` list although SR9–SR11 did not run. Preserve those files and correct them only in a derived namespace.
- SR7 did not implement its claimed structural alternative: both actual phase orders are XY→Z→orientation→descent→dwell; only step/clip/budget values differ. Do not rerun either identity.
- Do not start the old Teleop argv yet. The recorder defaults to gripper `+1`, exposes only a toggle, does not persist per-step raw contact/source/goal gates, and lacks the current attempt identity/finalizer lifecycle. There is no episode for human visual review now.
- The sole active plan is `docs/AXIS_PI05_TRACK3_SR_CLOSURE_AND_TASK10_USER_TELEOP_GOLD_CANARY_PLAN.md`. UT0–UT3 perform derived closure and recorder readiness with no simulator/GPU; after `ready_for_user_calibration`, the user may explicitly take over, calibrate without saving, and record at most one task10 candidate. Only machine success releases GPU3 exact replay `3/3` and a visual-review card. Accepted promotion, training/norm, FAST/ZIP, upload and submission remain prohibited.
- Post-lifecycle validation: SR strict JSON `27/27`, audit `2/2`, machine handoff `1/1`, targeted `18 passed`, full `425 passed, 7 skipped, 23 warnings`; py_compile/diff-check/no-partial/process boundaries passed. Public LIBERO and nested OpenPI are clean; the pre-existing `LIBERO-plus` dirty set is preserved. GPU0 has an external job and was not touched; GPU1–3 are idle.

## 25. Step-horizon gates vs official Axis 30k (2026-09-12, open design review)

User 2026-09-12: official Axis is about 30000 optimizer steps with good results, but local 100/250/no-auto-500 gates stopped continuation; the gate design may be too conservative. This is recorded for later high-level review, not as authorization.

- Machine note: `data/reports/axis_libero_30k/t1_t2_dedicated_panel_and_official_250_horizon_20260911/step_horizon_gate_design_review_20260912.json`
- Plan section: `docs/AXIS_PI05_T1_T2_DEDICATED_PANEL_AND_OFFICIAL_250_HORIZON_PLAN_20260911.md` §13
- Do not collapse: Axis init is not a finished LIBERO 30k post-train; the parked dense 30k plan failed on V100 OOM; canary100 T1=0.30 / T2=0.31 (user-reported) only freezes those two 100-step canaries. Do not start 250/500/1k/30k from this note, and do not resume canary100 `train_state` into a longer cosine.

## 26. Official Blackwell T2 H10K launch issues (2026-09-13)

User 2026-09-13 ~16:42 JST: official `t2_p30_r6_h10k_r1` started. Do not reconstruct the failure taxonomy from chat. Canonical write-up:

- `docs/AXIS_PI05_OFFICIAL_BLACKWELL_T2_H10K_LAUNCH_ISSUES_20260913.md`
- `data/reports/axis_libero_30k/t2_p30_r6_h10k_r1_official_launch_issues_20260913/summary.json`

Four independent classes, plus source-tree skew. Do not collapse them into “pip version mismatch”:

1. JAX stack wall: official GPU requires jax 0.11.1 + CUDA 13; OpenPI pins jax 0.5.3 / orbax 0.11.13. Local V100 green tests cannot see this.
2. Missing deps/cache: `gcsfs`, paligemma tokenizer file, uv symlink wrapper. Incomplete clean machine, not a minor bump.
3. On-disk layout: official IPEC tar is LeRobot v3 `file-*.parquet`; local mix assumed v2.1. Same pip as V100 still explodes. Hub `RevisionNotFoundError` was this, not `create_tag`.
4. Operational: empty run dir, paste, sleep-killed instance, no data-push.
5. Overlay `train.py` vs public clone HEAD `utils.py` (`TrainState.rng`) is file mix, not JAX.

Recipe/SHA were not the launch-failure cause. Leave the Jupyter tab; do not Stop My Server; data-push after checkpoints. Agent still does not upload or submit.

## 27. T2 H10K post-train agent handoff (2026-09-13)

User: after training, official server may have only ~10 hours left. Next agent must not reconstruct from chat.

- Contract: `docs/AXIS_PI05_T2_H10K_POST_TRAIN_AGENT_HANDOFF_20260913.md`
- JSON: `data/reports/axis_libero_30k/t2_p30_r6_h10k_r1_post_train_handoff_20260913/summary.json`

Official Blackwell remaining uses: data-push Orbax, fold on jax 0.11.1, persist safetensors SHA, optional restore smoke, resume same run_id only. Full C20 / FAST ZIP / submit / Fast400 / T1 / T3 / dense do not get the leftover official quota. New training identity stays false; auto-submit stays false.

Live ~17:10 JST: pid 6950, step≈1684, `examples_seen=64`, `micro_steps=16`, `valid_action_elements=640` constant, `temporal_padding_loss_mass=0`, `seven_d_loss_mass==loss_sum`. 640 is 16×4×10 from upstream `compute_loss` tensor size, not Axis valid7 4480. Do not stop the run for that. W&B scale vs last night is not apples-to-apples if last night had `compute_loss_components`.

## 28. T2 H10K tomorrow persist/fold playbook (2026-09-13 night)

User sequence for 2026-09-14 morning: read-only check → copy highest complete KEEP step via rsync staging → `parc-home-sync data-push` → fold on official trainvenv → Stop only after persist. Exact cells/commands:

- `docs/AXIS_PI05_T2_H10K_TOMORROW_PERSIST_FOLD_PLAYBOOK_20260914.md`
- `data/reports/axis_libero_30k/t2_p30_r6_h10k_r1_tomorrow_persist_fold_20260914/summary.json`

Fold entry is `trainvenv_python.sh -m tools.axis_pi05.convert_for_lerobot`. Probe torch/safetensors first; do not pip-install torch into trainvenv. Docker image name is README placeholder `parc2026`; run `docker images` on site. `parc-home-sync data-push` is documented as `~/data` only.

User 2026-09-14 morning (official Jupyter): training finished at step 10000; GPU empty. Persist+fold+data-push completed. Do not reconstruct from chat.

```text
last_complete_step             10000
train_state_in_persist         true
folded_safetensors_bytes       14467165872
folded_safetensors_sha256      bfc678f96190c922a84aa893af111db6ed7126047274257eee449993aa2d7cd3
norm_sha256                    39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41
can_stop_server                true after final pgrep/nvidia-smi empty check
```

Official tree had no `tools.axis_pi05.convert_for_lerobot`; fold used `work/axis_fold/run_fold.py` with `checkpoint_dir="pi05"` and a copy of `transformers_replace` into trainvenv (transformers 4.53.2, no pip reinstall). Next after Stop: local GPU3 C20 / FAST ZIP. Auto-submit stays false.

## 29. T2 H10K local transfer / HF 404 / GPU idle (2026-09-14)

User 2026-09-14 ~12:42 JST: keep the live S3 presigned download; do not switch to HF. GPU plan: GPU3 reserved for T2 C20 after SHA; no 4-way C20 split; T3 Q4/H300 only after T2 delivery, train on GPU1+GPU2; GPU0 idle; T1/dense off.

HF 404 is not a token bug. `HfApi.whoami()` is `Huangpenguin`. Private canary/R6 model repos exist. `Huangpenguin/parc2026-t2-p30-r6-h10k-r1-folded-10000` was never created (create_repo was blocked this morning). `--use-hf` against that id will 404 until a real private repo exists and the 14GB file is uploaded **from this machine after S3 completes**. Uploading from official Blackwell would compete with S3 and is slower. Do not interrupt curl PID 3587101.

Live ~12:43 JST: `model.safetensors.partial` ~3.7 GiB / 13.47 GiB (~27%); formal `model.safetensors` not yet renamed. Presigned URL expiry cited by user ~23:12 JST; remaining 4–5 h fits. After SHA `bfc678f9…` matches, GPU3 FAST/C20/ZIP only.
