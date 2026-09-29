from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.axis_pi05 import track3_paired_eval_finalizer_v2 as finalizer


ROOT = Path(__file__).parents[1]


def _source_bundle(tmp_path: Path, cases: dict[str, str]) -> tuple[dict[str, str], dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "comparator_source_sha256": tmp_path / "comparator.py",
        "worker_adapter_sha256": tmp_path / "worker_adapter.py",
        "worker_plan_sha256": tmp_path / "worker_plan.md",
        "comparison_plan_sha256": tmp_path / "comparison_plan.md",
    }
    for field, path in source_paths.items():
        path.write_text(f"{field}\n", encoding="utf-8")
    identity = {field: finalizer.sha256_file(path) for field, path in source_paths.items()}
    identity["input_identity_sha256"] = finalizer.input_identity_for_cases(cases)
    return identity, source_paths


def _cases() -> dict[str, str]:
    return {"case-0": "fp-0", "case-1": "fp-1"}


@pytest.fixture()
def valid_mp4(tmp_path: Path) -> Path:
    path = tmp_path / "valid.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:r=1:d=1",
            "-frames:v",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
    )
    return path


def _artifacts(tmp_path: Path, video: Path, *, all_zero: bool = False, caller_zero_waiver: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    traces = {role: [] for role in finalizer.PAYLOAD_NAMES}
    videos = {role: [] for role in finalizer.PAYLOAD_NAMES}
    values = np.zeros((3, 7), dtype=np.float32) if all_zero else np.ones((3, 7), dtype=np.float32)
    for role in finalizer.PAYLOAD_NAMES:
        for case_id, fingerprint in _cases().items():
            trace = tmp_path / f"{role}-{case_id}.npy"
            np.save(trace, values, allow_pickle=False)
            role_video = tmp_path / f"{role}-{case_id}.mp4"
            shutil.copyfile(video, role_video)
            trace_row = {"case_id": case_id, "case_fingerprint": fingerprint, "path": str(trace)}
            if caller_zero_waiver:
                trace_row["allow_zero"] = True
            traces[role].append(trace_row)
            videos[role].append({"case_id": case_id, "case_fingerprint": fingerprint, "path": str(role_video)})
    return traces, videos


def _worker_payload(role: str, identity: dict[str, str], cases: dict[str, str], *, status: str = "completed") -> dict:
    return {
        "schema_version": finalizer.WORKER_RESULT_SCHEMA,
        "status": status,
        "role": role,
        "identity": identity,
        "infrastructure_error": False,
        "cases": [
            {"case_id": case_id, "case_fingerprint": fingerprint, "status": "completed", "infrastructure_error": False}
            for case_id, fingerprint in cases.items()
        ],
    }


def _comparison(identity: dict[str, str], cases: dict[str, str], *, behavior: bool = True, retention: bool = True) -> dict:
    return {
        "schema_version": finalizer.COMPARISON_RESULT_SCHEMA,
        "status": "completed",
        "comparator_source_sha256": identity["comparator_source_sha256"],
        "comparison_plan_sha256": identity["comparison_plan_sha256"],
        "input_identity_sha256": identity["input_identity_sha256"],
        "zero_trace_policy": finalizer.ZERO_TRACE_POLICY,
        "behavior_gate": behavior,
        "retention_gate": retention,
        "cases": [{"case_id": case_id, "case_fingerprint": fingerprint} for case_id, fingerprint in cases.items()],
    }


def _setup(tmp_path: Path, video: Path):
    cases = _cases()
    identity, source_paths = _source_bundle(tmp_path / "sources", cases)
    traces, videos = _artifacts(tmp_path / "artifacts", video)
    worker_payloads = {role: _worker_payload(role, identity, cases) for role in finalizer.PAYLOAD_NAMES}
    comparison = _comparison(identity, cases)
    return cases, identity, source_paths, traces, videos, worker_payloads, comparison


def _finish(tmp_path: Path, video: Path, *, crash_after: str | None = None, identity=None, traces=None, videos=None, payloads=None, comparison=None):
    cases, expected, source_paths, default_traces, default_videos, default_payloads, default_comparison = _setup(tmp_path, video)
    worker = finalizer.FinalizerV2(tmp_path / "attempt", identity or expected, source_paths)
    return worker.finalize(
        payloads or default_payloads,
        cases=cases,
        traces=traces or default_traces,
        videos=videos or default_videos,
        comparison_result=comparison or default_comparison,
        crash_after=crash_after,
    )


def test_identity_fields_bind_to_explicit_sources_and_missing_or_changed_values_fail_closed(tmp_path: Path, valid_mp4: Path):
    cases = _cases()
    identity, source_paths = _source_bundle(tmp_path / "sources", cases)
    assert identity["worker_plan_sha256"] != identity["comparison_plan_sha256"]
    for field in finalizer.IDENTITY_FIELDS:
        missing = dict(identity)
        missing.pop(field)
        with pytest.raises(ValueError, match=field):
            finalizer.validate_identity(missing)
    for field in finalizer.SOURCE_IDENTITY_FIELDS:
        missing_paths = dict(source_paths)
        missing_paths.pop(field)
        with pytest.raises(finalizer.IdentityMismatchError):
            finalizer.FinalizerV2(tmp_path / f"missing-{field}", identity, missing_paths)
        changed = dict(identity)
        changed[field] = "f" * 64
        with pytest.raises(finalizer.IdentityMismatchError, match="source path"):
            finalizer.FinalizerV2(tmp_path / f"changed-{field}", changed, source_paths).finalize(
                {role: _worker_payload(role, changed, cases) for role in finalizer.PAYLOAD_NAMES},
                cases=cases,
                traces={role: [] for role in finalizer.PAYLOAD_NAMES},
                videos={role: [] for role in finalizer.PAYLOAD_NAMES},
                comparison_result={},
            )


def test_ffprobe_rejects_arbitrary_and_truncated_bytes(tmp_path: Path, valid_mp4: Path):
    assert finalizer.validate_mp4(valid_mp4)["valid"] is True
    arbitrary = tmp_path / "arbitrary.mp4"
    arbitrary.write_bytes(b"not a video")
    assert finalizer.validate_mp4(arbitrary)["valid"] is False
    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(valid_mp4.read_bytes()[:64])
    assert finalizer.validate_mp4(truncated)["valid"] is False


def test_trace_shape_finite_nonzero_and_zero_waiver_cannot_release(tmp_path: Path):
    valid = tmp_path / "valid.npy"
    np.save(valid, np.ones((2, 7), dtype=np.float32), allow_pickle=False)
    assert finalizer.validate_trace(valid, case_id="case", case_fingerprint="fp")["valid"] is True
    zero = tmp_path / "zero.npy"
    np.save(zero, np.zeros((2, 7), dtype=np.float32), allow_pickle=False)
    assert finalizer.validate_trace(zero, case_id="case", case_fingerprint="fp")["valid"] is False
    nan = tmp_path / "nan.npy"
    np.save(nan, np.full((2, 7), np.nan, dtype=np.float32), allow_pickle=False)
    assert finalizer.validate_trace(nan, case_id="case", case_fingerprint="fp")["valid"] is False
    wrong_shape = tmp_path / "wrong.npy"
    np.save(wrong_shape, np.ones((2, 6), dtype=np.float32), allow_pickle=False)
    assert finalizer.validate_trace(wrong_shape, case_id="case", case_fingerprint="fp")["valid"] is False


def test_opaque_or_malformed_worker_payload_forces_release_false(tmp_path: Path, valid_mp4: Path):
    cases, identity, source_paths, traces, videos, payloads, comparison = _setup(tmp_path / "opaque", valid_mp4)
    payloads["baseline"] = b"opaque worker bytes"
    result = finalizer.FinalizerV2(tmp_path / "opaque" / "attempt", identity, source_paths).finalize(
        payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison
    )
    assert result["release"] is False
    assert result["gates"]["coverage_gate"] is False

    duplicate = json.dumps({"status": "completed", "status": "completed"}).encode()
    payloads["baseline"] = duplicate
    result = finalizer.FinalizerV2(tmp_path / "duplicate" / "attempt", identity, source_paths).finalize(
        payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison
    )
    assert result["release"] is False


def test_missing_case_or_infrastructure_worker_result_is_not_coverage(tmp_path: Path, valid_mp4: Path):
    cases, identity, source_paths, traces, videos, payloads, comparison = _setup(tmp_path, valid_mp4)
    payloads["candidate"]["cases"] = payloads["candidate"]["cases"][:-1]
    payloads["baseline"]["infrastructure_error"] = True
    result = finalizer.FinalizerV2(tmp_path / "attempt", identity, source_paths).finalize(
        payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison
    )
    assert result["release"] is False
    assert result["gates"]["coverage_gate"] is False
    assert result["gates"]["identity_gate"] is True


def test_first_payload_recovery_is_exactly_once(tmp_path: Path, valid_mp4: Path):
    cases, identity, source_paths, traces, videos, payloads, comparison = _setup(tmp_path, valid_mp4)
    worker = finalizer.FinalizerV2(tmp_path / "attempt", identity, source_paths)
    with pytest.raises(finalizer.InjectedCrash):
        worker.finalize(payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison, crash_after="first_payload")
    assert (tmp_path / "attempt" / "staging.partial" / "payloads" / "baseline.json").is_file()
    result = worker.finalize(payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison)
    assert result["release"] is True
    with pytest.raises(FileExistsError, match="immutable"):
        worker.finalize(payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison)


@pytest.mark.parametrize("phase", ["paired_summary", "release_decision", "manifest"])
def test_summary_release_manifest_crash_points_resume(tmp_path: Path, valid_mp4: Path, phase: str):
    cases, identity, source_paths, traces, videos, payloads, comparison = _setup(tmp_path, valid_mp4)
    worker = finalizer.FinalizerV2(tmp_path / "attempt", identity, source_paths)
    with pytest.raises(finalizer.InjectedCrash):
        worker.finalize(payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison, crash_after=phase)
    result = worker.finalize(payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison)
    assert result["status"] == "completed"
    assert finalizer.validate_published_attempt(tmp_path / "attempt")["release"] is True


def test_rename_crash_leaves_valid_completed_attempt_and_rejects_overwrite(tmp_path: Path, valid_mp4: Path):
    cases, identity, source_paths, traces, videos, payloads, comparison = _setup(tmp_path, valid_mp4)
    worker = finalizer.FinalizerV2(tmp_path / "attempt", identity, source_paths)
    with pytest.raises(finalizer.InjectedCrash):
        worker.finalize(payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison, crash_after="rename")
    assert finalizer.validate_published_attempt(tmp_path / "attempt")["release"] is True
    with pytest.raises(FileExistsError):
        worker.finalize(payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison)


def test_changed_identity_cannot_resume_staging(tmp_path: Path, valid_mp4: Path):
    cases, identity, source_paths, traces, videos, payloads, comparison = _setup(tmp_path, valid_mp4)
    worker = finalizer.FinalizerV2(tmp_path / "attempt", identity, source_paths)
    with pytest.raises(finalizer.InjectedCrash):
        worker.finalize(payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison, crash_after="first_payload")
    changed = dict(identity)
    changed["input_identity_sha256"] = "f" * 64
    with pytest.raises(finalizer.IdentityMismatchError):
        finalizer.FinalizerV2(tmp_path / "attempt", changed, source_paths).finalize(
            payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison
        )


def test_manifest_duplicate_omission_and_hash_mismatch_are_rejected(tmp_path: Path, valid_mp4: Path):
    cases, identity, source_paths, traces, videos, payloads, comparison = _setup(tmp_path, valid_mp4)
    worker = finalizer.FinalizerV2(tmp_path / "attempt", identity, source_paths)
    worker.finalize(payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison)
    manifest_path = tmp_path / "attempt" / "completed" / "complete_set_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["payloads"].append(dict(manifest["payloads"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(finalizer.AttemptStateError, match="exactly once"):
        finalizer.validate_published_attempt(tmp_path / "attempt")

    cases, identity, source_paths, traces, videos, payloads, comparison = _setup(tmp_path / "missing", valid_mp4)
    worker = finalizer.FinalizerV2(tmp_path / "missing" / "attempt", identity, source_paths)
    worker.finalize(payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison)
    path = tmp_path / "missing" / "attempt" / "completed" / "complete_set_manifest.json"
    data = json.loads(path.read_text())
    data["payloads"] = data["payloads"][:1]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(finalizer.AttemptStateError, match="exactly once"):
        finalizer.validate_published_attempt(tmp_path / "missing" / "attempt")

    cases, identity, source_paths, traces, videos, payloads, comparison = _setup(tmp_path / "hash", valid_mp4)
    worker = finalizer.FinalizerV2(tmp_path / "hash" / "attempt", identity, source_paths)
    worker.finalize(payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison)
    path = tmp_path / "hash" / "attempt" / "completed" / "complete_set_manifest.json"
    data = json.loads(path.read_text())
    data["payloads"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(finalizer.AttemptStateError, match="hash"):
        finalizer.validate_published_attempt(tmp_path / "hash" / "attempt")


def test_cross_links_and_gate_agreement_are_rechecked(tmp_path: Path, valid_mp4: Path):
    cases, identity, source_paths, traces, videos, payloads, comparison = _setup(tmp_path, valid_mp4)
    worker = finalizer.FinalizerV2(tmp_path / "attempt", identity, source_paths)
    worker.finalize(payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison)
    root = tmp_path / "attempt" / "completed"
    release_path = root / "release_decision.json"
    manifest_path = root / "complete_set_manifest.json"
    release = json.loads(release_path.read_text())
    release["paired_summary_sha256"] = "0" * 64
    release_path.write_text(json.dumps(release), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text())
    manifest["release_decision_sha256"] = hashlib.sha256(release_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(finalizer.AttemptStateError, match="release-to-paired"):
        finalizer.validate_published_attempt(tmp_path / "attempt")

    cases, identity, source_paths, traces, videos, payloads, comparison = _setup(tmp_path / "gates", valid_mp4)
    worker = finalizer.FinalizerV2(tmp_path / "gates" / "attempt", identity, source_paths)
    worker.finalize(payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison)
    root = tmp_path / "gates" / "attempt" / "completed"
    release_path = root / "release_decision.json"
    manifest_path = root / "complete_set_manifest.json"
    release = json.loads(release_path.read_text())
    release["gates"]["behavior_gate"] = False
    release_path.write_text(json.dumps(release), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text())
    manifest["release_decision_sha256"] = hashlib.sha256(release_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(finalizer.AttemptStateError, match="gates disagree"):
        finalizer.validate_published_attempt(tmp_path / "gates" / "attempt")


def test_comparison_gate_is_not_a_bare_caller_boolean(tmp_path: Path, valid_mp4: Path):
    cases, identity, source_paths, traces, videos, payloads, _ = _setup(tmp_path, valid_mp4)
    result = finalizer.FinalizerV2(tmp_path / "attempt", identity, source_paths).finalize(
        payloads,
        cases=cases,
        traces=traces,
        videos=videos,
        comparison_result={"behavior_gate": True, "retention_gate": True},
    )
    assert result["release"] is False
    assert result["gates"]["behavior_gate"] is False


def test_zero_trace_cannot_be_waived_by_caller(tmp_path: Path, valid_mp4: Path):
    cases, identity, source_paths, traces, videos, payloads, comparison = _setup(tmp_path, valid_mp4)
    traces, videos = _artifacts(tmp_path / "zero-artifacts", valid_mp4, all_zero=True, caller_zero_waiver=True)
    result = finalizer.FinalizerV2(tmp_path / "attempt", identity, source_paths).finalize(
        payloads, cases=cases, traces=traces, videos=videos, comparison_result=comparison
    )
    assert result["release"] is False
    assert result["gates"]["trace_gate"] is False


def test_old_namespace_fingerprints_are_unchanged(tmp_path: Path, valid_mp4: Path):
    historical = [
        ROOT / "data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002/paired_summary.json",
        ROOT / "data/reports/axis_libero_30k/t3_fast400_h300_step100_salvage_20260915_attempt_0001/paired_summary.json",
        ROOT / "data/reports/axis_libero_30k/t3_post_salvage_failure_analysis_20260916_attempt_0001/summary.json",
    ]
    assert all(path.is_file() for path in historical)
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in historical}
    _finish(tmp_path, valid_mp4)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in historical}
    assert before == after
