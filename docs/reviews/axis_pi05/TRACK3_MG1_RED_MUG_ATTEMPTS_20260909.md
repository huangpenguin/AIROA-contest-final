# Track3 MG1 red-mug attempt closeout (2026-09-09)

This is an execution closeout, not an independent audit.

## Result

MG1 ended with family_failure for both authorized red-mug attempts:

- Assisted: fresh continuous MuJoCo from the frozen 314-frame start state;
  77 frames; acquisition window not observed; complete telemetry; rejected.
- Manual: 4 Hz no-save calibration passed; formal recorder stopped at 300/300;
  source remained true, goal remained false, success was false; rejected.

No episode entered pending, no replay ran, and no human review card exists.
Red-mug quotas are exhausted: assisted 1/1 and manual 1/1.

## Evidence

- data/reports/axis_libero_30k/track3_operations_mg1_red_mug_assisted_20260909/
  preserves the pre-physics ModuleNotFoundError: libero launch failure.
- data/reports/axis_libero_30k/track3_operations_mg1_red_mug_assisted_retry_20260909/
  contains the code-only retry, assisted result, 4 Hz canary, manual result,
  artifact hashes, manifest, and closure.
- Rejected manual episode:
  data/reports/axis_libero_30k/track3_reverse_pilot_v2/rejected/official_t3_red_mug_1788934804/.
- Historical 314-frame seed SHA set remained unchanged.

O2, training, candidate ZIP, upload, and submission remain not_evaluated.
The next stop is MG2_alphabet_soup_requires_explicit_user_takeover.
