# Track3 O1.0 red-mug read-only retiming — 2026-09-08

## Result

O1.0 is complete and hard-stopped at `ready_for_native_vnc_no_save_canary`.
The preferred input was the 388-frame historical red-mug episode
`official_t3_red_mug_1788838632`; the 468-frame episode
`official_t3_red_mug_1788838293` was read as reference only.

No historical raw or metadata was changed. No replay, GPU, MuJoCo, Teleop,
new recording, Infinity reversal, or Scenic augmentation was started.

## Derived keyposes

The keyposes are planning landmarks, not a promoted trajectory:

| keypose | 388-step source frame | derivation |
| --- | ---: | --- |
| approach | 145 | last `precontact_xy` frame before the selected stable grasp dwell |
| grasp | 146–156 | final stable `grasp_dwell`, source true and goal false |
| lift | 157 | first `source_predicate=false` frame |
| transport | 278 | EEF clearance apex before the first goal predicate |
| place | 365 | first `goal_predicate=true` frame |
| release | 373 | post-place gripper transition to OPEN |

The historical contact arrays remain diagnostic only: the 388-step input has
`contact_n=42–52` with capacity 32 and all 388 frames truncated. The 468-step
reference has `contact_n=41–54` with capacity 32 and all 468 frames truncated.
Neither input is gold-eligible or replay-evaluated.

## Goal-first planning contract

The derived blueprint targets approach 80, grasp 91, lift 102, transport 160,
place 220, release 230, and settle 240. This keeps the planned goal before
the 240-frame goal-first budget, leaves 20 frames inside the 260-frame total
target, and remains below the hard 300-frame limit. These are retiming targets
only; no action sequence was generated.

The future recording must use the v4 writer contract: episode-relative paths,
atomic final artifacts, typed flat contact storage with offsets, and
`saved_contact_n == contact_n` for every frame. Acquisition must be a forward
close/grasp-dwell gripper↔subject event before source release; goal-table
contact and a final-frame backward inference cannot satisfy it.

## Evidence

Execution commit: `bef338d5885ef0e0d3355784a0ac6780dd90ffa0`.

- Machine namespace: `data/reports/axis_libero_30k/track3_operations_o1_0_red_mug_retiming_20260908/`
- Summary: `summary.json`
- Input/hash lock: `input_lock.json`
- Keypose report: `keypose_report.json`
- Retiming blueprint: `retiming_blueprint.json`
- Unique canonical attempt lock: `canonical_attempt_lock.json`
- Validation argv and non-actions: `validation_commands.json`

The next authorized action is the user-online native MuJoCo/TurboVNC no-save
canary. O2 remains frozen until an O1 canonical gold is accepted with fresh
replay `3/3`.
