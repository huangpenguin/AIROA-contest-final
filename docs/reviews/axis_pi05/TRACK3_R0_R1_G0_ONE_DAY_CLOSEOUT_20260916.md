# Track3 R0/R1/G0 one-day closeout

> Date: 2026-09-16 (Asia/Tokyo)  
> Scope: finalizer remediation, frozen-artifact reject integration, and read-only lineage feasibility  
> Decision: `D2_unreachable_within_one_day`

## Result

R0 remediation is complete. The finalizer now fails closed on duplicate payload
manifest entries, broken release/paired links, contradictory gates, opaque or
unbound worker payloads, missing source-byte bindings, non-strict JSON, invalid
MP4 bytes, and caller attempts to waive all-zero traces. Published attempts are
replayed from their real files and recover atomically after each injected crash
point.

The scoped focused test file collected 16 tests and exited successfully.
`py_compile` and `git diff --check` also passed. The remediated finalizer source
SHA is `3e538a8f9289945f7fc11f842ceeb15a2ced98269e5b0f26a46609ada56aa7d7`.

R1 translated the frozen legacy R6 and step100 case records into the strict
worker envelope in a new namespace. It reused the existing trace/video files;
it did not start a simulator, policy server, evaluator worker, or GPU. The
result was:

| gate/result | value |
|---|---:|
| reverse baseline → candidate | `0/4 → 0/4` |
| C20 baseline → candidate | `15/20 → 14/20` |
| coverage / identity / trace / video | all pass |
| behavior gate | fail |
| release | `false` |

The two frozen paired-summary SHA values were unchanged before and after R1.

G0 inspected all eight accepted reverse episodes. All declared artifact hashes
match the bytes on disk, but no accepted episode contains an explicit
recorder/session/supervisor/process identity or an episode-bound event log. All
eight `promotion_source` pointers target missing pending directories. Distinct
episode IDs, timestamps, start-state hashes, trajectory hashes, 3/3 replay, and
human review are therefore retained as artifact evidence only—not promoted to
independent lineage.

| task | accepted | explicit independent lineages | reverse dev evidence |
|---|---:|---:|---:|
| alphabet soup | 2 | 0 | no |
| red mug | 2 | 0 | no |
| two mugs | 1 | 0 | no |
| yellow book | 3 | 0 | no |

Because the data gate requires every task to have at least three accepted
episodes, two proven independent lineages, and lineage-disjoint reverse dev,
G1/D3/D4/D5/D6 are cancelled for this deadline. No mix, norm, training,
behavior evaluation, fold, FAST, ZIP, upload, or submission was performed.

## Evidence

- R1 machine summary: `data/reports/axis_libero_30k/t3_d1_finalizer_r1_reject_integration_20260916_attempt_0001/summary.json`
- G0 machine summary: `data/reports/axis_libero_30k/t3_g0_lineage_feasibility_audit_20260916_attempt_0001/summary.json`
- G0 episode table: `data/reports/axis_libero_30k/t3_g0_lineage_feasibility_audit_20260916_attempt_0001/lineage_episode_table.md`
- finalizer: `tools/axis_pi05/track3_paired_eval_finalizer_v2.py`
- R1 adapter: `tools/axis_pi05/run_track3_d1_finalizer_r1_reject_integration.py`
- G0 auditor: `tools/axis_pi05/audit_track3_g0_lineage_feasibility.py`

## Submission posture

Keep the known Track1 R6 score `0.47` and Track2 R6 score `0.33`. Do not use
the failed H10K result. Do not publish the failed Track3 H300 step100/step300
candidate. Track3 R6 remains a mechanically usable but known-zero fallback
only if a submission form forces a Track3 artifact and the user explicitly
chooses it.
