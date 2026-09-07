# Track 3 SG0–SG6 execution closeout

Date: 2026-09-07 (Asia/Tokyo)

This is an execution record, not an independent audit. The CR audit and all
CR artifacts remain immutable historical inputs.

## Result

SG0–SG4 passed in the new namespace on `attempt_0003`. SG5 passed its live
clean-source release gate at root commit `d5c319b055431d2ec46681775438c91ad64a2be0`.
The first SG6 launch was fail-closed in `attempt_0004` as a shared
infrastructure failure: the child raised
`TypeError: compose_object_to_eef() got an unexpected keyword argument
object_position_offset_m` before creating the simulator environment.

The resulting SG6 status is therefore `hard_stop` / `infrastructure_failure`.
It is not a family failure, not a behavior result, and not a success. SG7,
SG8, SG9, SG10, SG11 and SG12 were not evaluated by physics; no second
identity was launched.

The code defect was fixed afterward in commit
`3e156dd` and covered by an additional regression. This commit has not been
used to re-release SG5 or rerun SG6, so the original SG6 hard stop remains
unchanged and valid.

## Evidence

- SG implementation: `tools/axis_pi05/track3_contact_truth_and_support_gold_demo.py`
- SG regression: `tests/test_track3_contact_truth_and_support_gold_demo.py`
- SG evidence namespace:
  `data/reports/axis_libero_30k/track3_contact_truth_and_support_gold_demo/`
- Failed launch: `attempts/attempt_0004/summary.json`
- SG6 child error:
  `attempts/attempt_0004/canary/child_result.json`
- SG0–SG5 successful code-only attempt: `attempts/attempt_0003/`

The SG4 recorder command was validated as:

```text
.venv_axis_pi05/bin/python -m tools.axis_pi05.track3_public_teleop --record --task-index 10 --operator operator-local --preview-port 8765
```

No recording, replay, accepted promotion, generation, training, ZIP, upload,
or submission was performed.

## Validation and boundaries

- SG-specific regression: `11 passed`
- Full test suite after the fix: `399 passed, 7 skipped, 23 warnings`
- `py_compile` and `git diff --check`: passed
- SG JSON strict scan: `37` files, duplicate keys rejected
- SG `*.partial` residue: `0`
- GPU compute processes: none; GPU0–GPU3 untouched by SG6
- Root and OpenPI worktrees: clean; pre-existing `LIBERO-plus` dirty state preserved
- External YOLO process observed and not touched

## Required next authority

Because the plan mandates an immediate sticky global hard stop on an
infrastructure failure, the corrected `3e156dd` code must not silently consume
another identity. A new reviewed release decision is required before any
fresh SG5 gate or SG6 physics launch.
