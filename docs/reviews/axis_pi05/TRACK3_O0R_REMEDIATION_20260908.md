# Track3 O0R remediation review — 2026-09-08

Audit class: approved

This report is a derived correction to the earlier O0 review. The earlier
remediation_required result remains historical and is not rewritten.

## Frozen inputs and boundary

- Root HEAD before remediation: bb0793f5f8558115d2fc20b07044c252d2c50efa.
- Nested third_party/axis_openpi HEAD: b58aff0750f7a9ee07f3feb5df951a20ad2c76e1.
- Raw inventory: 69 files, SHA-256
  3a385f0a56a58c9082d74cc33f1a1f7552401b3eb0caa1e98f5c22399fff86b1.
- The two supplied PDFs and the strategy draft are hashed in
  data/reports/axis_libero_30k/track3_operations_o0r_remediation_20260908/summary.json.
- No Track3/Teleop/MuJoCo process was active. GPU0–2 remained externally
  occupied and GPU3 was unused. No replay, GPU work, Teleop, or recording was
  performed.

## Historical raw disposition

Historical metadata and raw episode files were not modified. Derived path
resolution for every metadata-bearing episode is persisted in
path_resolution.json; each expected final artifact has a relative final
name, byte count, SHA-256, expected_sha_match=true, and
historical_metadata_unchanged=true.

Both pending episodes remain explicitly ineligible:

- official_t3_red_mug_1788838293: 468 frames, contact_n 41–54, capacity 32,
  468/468 truncated, acquisition barrier 467.
- official_t3_red_mug_1788838632: 388 frames, contact_n 42–52, capacity 32,
  388/388 truncated, acquisition barrier 387.

Each derived eligibility report records:
status=pending_historical_raw, replay=not_evaluated,
gold_eligible=false, and
reason=[over_budget, contact_raw_incomplete, acquisition_gate_invalid].
The 388-frame episode is marked preferred_seed=true; the 468-frame episode
is reference-only.

## Future-writer remediation

track3_public_teleop.py now emits schema track3_reverse_pilot_v4:

- metadata stores trajectory.npz, front.mp4, and wrist.mp4 as
  episode-relative paths only; absolute, parent traversal, and .partial
  references are rejected;
- contact names use a typed flat table with contact_offsets; every archive
  verifies diff(contact_offsets) == contact_n, with no fixed 32-contact cap;
- acquisition is a forward-only first stable gripper↔subject window during
  close/grasp dwell before source release. Goal-table contact and final-frame
  backward inference cannot satisfy it.

## Validation

- Targeted: 12 passed.
- Full: 434 passed, 7 skipped, 26 warnings.
- py_compile: passed.
- git diff --check: passed.
- Atomic move/clone, path rejection, 54-contact preservation, and late/goal
  contact rejection are covered by regression tests.

Machine evidence is under
data/reports/axis_libero_30k/track3_operations_o0r_remediation_20260908/.
