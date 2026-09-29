"""Fail-closed finalizer for future Track3 paired evaluation releases.

The historical H300 comparators are deliberately not imported here.  This
module is a release-authority boundary only: it validates worker JSON, binds
source identities to real files, rechecks traces/videos and publishes one
attempt directory atomically.  It never starts a worker, simulator, server or
GPU process.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "track3_paired_eval_finalizer_v2"
WORKER_RESULT_SCHEMA = "track3_worker_result_v2"
COMPARISON_RESULT_SCHEMA = "track3_comparison_result_v2"
ZERO_TRACE_POLICY = "nonzero_all_traces"
IDENTITY_FIELDS = (
    "comparator_source_sha256",
    "worker_adapter_sha256",
    "worker_plan_sha256",
    "comparison_plan_sha256",
    "input_identity_sha256",
)
SOURCE_IDENTITY_FIELDS = IDENTITY_FIELDS[:4]
PAYLOAD_NAMES = ("baseline", "candidate")
STAGING_NAME = "staging.partial"
COMPLETED_NAME = "completed"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FinalizerV2Error(RuntimeError):
    """Base error for a fail-closed finalizer operation."""


class IdentityMismatchError(FinalizerV2Error):
    """An identity or source binding differs from the frozen attempt."""


class AttemptStateError(FinalizerV2Error):
    """An attempt contains an invalid or unexpected artifact set."""


class InjectedCrash(FinalizerV2Error):
    """Test-only interruption at a named atomic-publication boundary."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)


def _strict_json_bytes(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise AttemptStateError(f"existing artifact differs and cannot be overwritten: {path}")
    _atomic_bytes(path, payload)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _canonical_json(value) + b"\n")


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def validate_identity(identity: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(identity, Mapping):
        raise ValueError("identity must be a mapping")
    return {field: _require_sha(identity.get(field), field) for field in IDENTITY_FIELDS}


def identity_sha256(identity: Mapping[str, Any]) -> str:
    return sha256_bytes(_canonical_json(validate_identity(identity)))


def _normalise_cases(cases: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, str]:
    if isinstance(cases, Mapping):
        result = {str(case_id): str(fingerprint) for case_id, fingerprint in cases.items()}
    else:
        result = {}
        for row in cases:
            if not isinstance(row, Mapping):
                raise ValueError("case rows must be mappings")
            case_id = row.get("case_id")
            fingerprint = row.get("case_fingerprint")
            if not isinstance(case_id, str) or not case_id or not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError("case rows require case_id and case_fingerprint")
            if case_id in result:
                raise ValueError(f"duplicate case_id: {case_id}")
            result[case_id] = fingerprint
    if not result or any(not case_id or not fingerprint for case_id, fingerprint in result.items()):
        raise ValueError("case set must contain non-empty identities")
    return result


def input_identity_for_cases(cases: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> str:
    case_map = _normalise_cases(cases)
    return sha256_bytes(_canonical_json({"cases": [{"case_id": k, "case_fingerprint": case_map[k]} for k in sorted(case_map)]}))


def _path_from_row(row: Mapping[str, Any]) -> Path | None:
    value = row.get("path")
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def validate_trace(path: Path | None, *, case_id: str, case_fingerprint: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_id": case_id,
        "case_fingerprint": case_fingerprint,
        "path": str(path) if path is not None else None,
        "shape": None,
        "dtype": None,
        "finite": False,
        "nonzero": False,
        "valid": False,
    }
    if path is None or not path.is_file():
        return result
    result["sha256"] = sha256_file(path)
    try:
        array = np.load(path, allow_pickle=False)
        result["shape"] = list(array.shape) if isinstance(array, np.ndarray) else None
        result["dtype"] = str(array.dtype) if isinstance(array, np.ndarray) else None
        if not isinstance(array, np.ndarray) or array.dtype.kind == "O" or array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 7:
            return result
        result["finite"] = bool(np.isfinite(array).all())
        result["nonzero"] = bool(np.any(array != 0))
        result["valid"] = bool(result["finite"] and result["nonzero"])
    except (OSError, ValueError, FloatingPointError, AttributeError, TypeError):
        return result
    return result


def validate_mp4(path: Path | None, *, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path) if path is not None else None,
        "bytes": 0,
        "ffprobe_returncode": None,
        "video_stream": False,
        "codec_name": None,
        "width": None,
        "height": None,
        "nb_frames": None,
        "duration": None,
        "valid": False,
    }
    if path is None or not path.is_file():
        return result
    result["bytes"] = path.stat().st_size
    if result["bytes"] <= 0:
        return result
    result["sha256"] = sha256_file(path)
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return result
    result["ffprobe_returncode"] = completed.returncode
    if completed.returncode != 0:
        return result
    try:
        parsed = json.loads(completed.stdout, object_pairs_hook=_reject_duplicate_keys)
        streams = parsed.get("streams")
        stream = streams[0] if isinstance(streams, list) and streams and isinstance(streams[0], Mapping) else None
        if stream is None or stream.get("codec_type") != "video":
            return result
        result["video_stream"] = True
        result["codec_name"] = stream.get("codec_name")
        result["width"] = int(stream.get("width", 0))
        result["height"] = int(stream.get("height", 0))
        raw_frames = stream.get("nb_frames")
        try:
            result["nb_frames"] = int(raw_frames) if raw_frames not in (None, "N/A") else None
        except (TypeError, ValueError):
            result["nb_frames"] = None
        raw_duration = stream.get("duration")
        if raw_duration in (None, "N/A") and isinstance(parsed.get("format"), Mapping):
            raw_duration = parsed["format"].get("duration")
        try:
            result["duration"] = float(raw_duration) if raw_duration not in (None, "N/A") else None
        except (TypeError, ValueError):
            result["duration"] = None
    except (TypeError, ValueError, KeyError, IndexError):
        return result
    duration_ok = isinstance(result["duration"], (int, float)) and math.isfinite(result["duration"]) and result["duration"] > 0
    frames_ok = isinstance(result["nb_frames"], int) and result["nb_frames"] > 0
    result["valid"] = bool(
        result["video_stream"]
        and isinstance(result["codec_name"], str)
        and bool(result["codec_name"])
        and isinstance(result["width"], int)
        and isinstance(result["height"], int)
        and result["width"] > 0
        and result["height"] > 0
        and (duration_ok or frames_ok)
    )
    return result


def _validate_artifact_bundle(
    bundle: Mapping[str, Any] | None,
    cases: Mapping[str, str],
    validator: Any,
) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    results: dict[str, list[dict[str, Any]]] = {role: [] for role in PAYLOAD_NAMES}
    complete = isinstance(bundle, Mapping) and set(bundle) == set(PAYLOAD_NAMES)
    if not complete:
        return results, False
    for role in PAYLOAD_NAMES:
        rows = bundle.get(role)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            complete = False
            continue
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                complete = False
                continue
            case_id = row.get("case_id")
            case_fingerprint = row.get("case_fingerprint")
            if not isinstance(case_id, str) or case_id not in cases or case_fingerprint != cases.get(case_id) or case_id in seen:
                complete = False
                continue
            seen.add(case_id)
            if "allow_zero" in row:
                complete = False
            path = _path_from_row(row)
            checked = validator(path, case_id=case_id, case_fingerprint=case_fingerprint) if validator is validate_trace else validator(path)
            # Keep the identity columns in both the initial and replayed
            # summaries.  Without these fields, replay cannot recover the
            # case key from a video row and would incorrectly reject every
            # otherwise valid published attempt.
            checked["case_id"] = case_id
            checked["case_fingerprint"] = case_fingerprint
            checked["role"] = role
            results[role].append(checked)
        if seen != set(cases):
            complete = False
    valid = complete and all(item.get("valid") is True for rows in results.values() for item in rows)
    return results, bool(valid)


def _artifact_summary(trace_results: Mapping[str, Sequence[Mapping[str, Any]]], video_results: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    return {
        "traces": {role: [dict(item) for item in rows] for role, rows in trace_results.items()},
        "videos": {role: [dict(item) for item in rows] for role, rows in video_results.items()},
    }


def _payload_bytes(value: bytes | bytearray | Mapping[str, Any] | Path | str) -> bytes:
    if isinstance(value, Mapping):
        return _canonical_json(value) + b"\n"
    if isinstance(value, (Path, str)):
        return Path(value).read_bytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise TypeError("payloads must be strict JSON mappings, bytes or filesystem paths")


def _worker_check(path: Path, role: str, identity: Mapping[str, str], cases: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": role,
        "status": None,
        "case_ids": [],
        "identity_matches": False,
        "infrastructure_error": None,
        "valid": False,
    }
    try:
        payload = strict_json_load(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return result
    if not isinstance(payload, Mapping):
        return result
    result["status"] = payload.get("status")
    result["infrastructure_error"] = payload.get("infrastructure_error")
    result["identity_matches"] = payload.get("identity") == dict(identity)
    rows = payload.get("cases")
    if not isinstance(rows, list):
        return result
    seen: set[str] = set()
    row_valid = True
    for row in rows:
        if not isinstance(row, Mapping):
            row_valid = False
            continue
        case_id = row.get("case_id")
        result["case_ids"].append(case_id)
        if not isinstance(case_id, str) or case_id in seen or case_id not in cases or row.get("case_fingerprint") != cases.get(case_id):
            row_valid = False
            continue
        seen.add(case_id)
        if row.get("status") != "completed" or row.get("infrastructure_error") is not False:
            row_valid = False
    result["case_ids"] = sorted(str(case_id) for case_id in result["case_ids"])
    result["valid"] = bool(
        payload.get("schema_version") == WORKER_RESULT_SCHEMA
        and payload.get("status") == "completed"
        and payload.get("role") == role
        and payload.get("infrastructure_error") is False
        and result["identity_matches"]
        and row_valid
        and seen == set(cases)
        and len(rows) == len(cases)
    )
    return result


def _read_comparison(value: Mapping[str, Any] | Path | str | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (Path, str)):
        try:
            parsed = strict_json_load(Path(value))
            return dict(parsed) if isinstance(parsed, Mapping) else {"_invalid": "comparison result is not an object"}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return {"_invalid": "comparison result is not strict JSON"}
    return {"_invalid": "comparison result is missing"}


def _comparison_check(comparison: Mapping[str, Any], identity: Mapping[str, str], cases: Mapping[str, str]) -> bool:
    rows = comparison.get("cases")
    if not isinstance(rows, list):
        return False
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in seen or case_id not in cases or row.get("case_fingerprint") != cases.get(case_id):
            return False
        seen.add(case_id)
    return bool(
        comparison.get("schema_version") == COMPARISON_RESULT_SCHEMA
        and comparison.get("status") == "completed"
        and comparison.get("comparator_source_sha256") == identity["comparator_source_sha256"]
        and comparison.get("comparison_plan_sha256") == identity["comparison_plan_sha256"]
        and comparison.get("input_identity_sha256") == identity["input_identity_sha256"]
        and (comparison.get("behavior_gate") is True or comparison.get("behavior_gate") is False)
        and (comparison.get("retention_gate") is True or comparison.get("retention_gate") is False)
        and comparison.get("zero_trace_policy") == ZERO_TRACE_POLICY
        and seen == set(cases)
        and len(rows) == len(cases)
    )


def _validate_source_bindings(identity: Mapping[str, str], source_paths: Mapping[str, str]) -> None:
    if set(source_paths) != set(SOURCE_IDENTITY_FIELDS):
        raise IdentityMismatchError("all four explicit source paths are required")
    for field in SOURCE_IDENTITY_FIELDS:
        path = Path(source_paths[field])
        if not path.is_file():
            raise IdentityMismatchError(f"missing explicit source path for {field}: {path}")
        actual = sha256_file(path)
        if actual != identity[field]:
            raise IdentityMismatchError(f"{field} does not match the bytes at its explicit source path")


def _file_set(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _directory_set(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()}


def _validate_staging_files(staging: Path, *, require_complete: bool) -> bool:
    allowed = {
        "attempt.lock.json",
        "payloads/baseline.json",
        "payloads/candidate.json",
        "paired_summary.json",
        "release_decision.json",
        "complete_set_manifest.json",
    }
    actual = _file_set(staging)
    required = {"attempt.lock.json"}
    if require_complete:
        required |= {"payloads/baseline.json", "payloads/candidate.json", "paired_summary.json", "release_decision.json", "complete_set_manifest.json"}
    return actual <= allowed and _directory_set(staging) <= {"payloads"} and required <= actual


def _request_fingerprint(
    identity: Mapping[str, str],
    source_paths: Mapping[str, str],
    cases: Mapping[str, str],
    payload_hashes: Mapping[str, str],
    comparison: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> str:
    request = {
        "identity": dict(identity),
        "source_paths": dict(source_paths),
        "cases": dict(cases),
        "payload_hashes": dict(payload_hashes),
        "comparison": dict(comparison),
        "artifacts": artifacts,
    }
    return sha256_bytes(_canonical_json(request))


def _derive_gates(
    *,
    comparison: Mapping[str, Any],
    comparison_valid: bool,
    worker_checks: Mapping[str, Mapping[str, Any]],
    trace_gate: bool,
    video_gate: bool,
    source_valid: bool,
    input_identity_valid: bool,
    complete_set_gate: bool,
) -> dict[str, bool]:
    worker_identity_valid = all(check.get("identity_matches") is True for check in worker_checks.values()) and set(worker_checks) == set(PAYLOAD_NAMES)
    return {
        "behavior_gate": bool(comparison_valid and comparison.get("behavior_gate") is True),
        "retention_gate": bool(comparison_valid and comparison.get("retention_gate") is True),
        "coverage_gate": bool(set(worker_checks) == set(PAYLOAD_NAMES) and all(check.get("valid") is True for check in worker_checks.values())),
        "trace_gate": bool(trace_gate),
        "video_gate": bool(video_gate),
        "identity_gate": bool(source_valid and input_identity_valid and worker_identity_valid),
        "complete_set_gate": bool(complete_set_gate),
    }


def _payload_descriptors(staging: Path, payload_hashes: Mapping[str, str]) -> list[dict[str, Any]]:
    descriptors = []
    for name in PAYLOAD_NAMES:
        path = staging / "payloads" / f"{name}.json"
        descriptors.append(
            {
                "name": name,
                "path": f"payloads/{name}.json",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "expected_sha256": payload_hashes[name],
            }
        )
    return descriptors


def _artifact_rows_from_summary(summary: Mapping[str, Any], section: str) -> dict[str, list[dict[str, Any]]]:
    value = summary.get("artifacts", {}).get(section) if isinstance(summary.get("artifacts"), Mapping) else None
    if not isinstance(value, Mapping):
        return {role: [] for role in PAYLOAD_NAMES}
    return {role: list(value.get(role, [])) if isinstance(value.get(role), list) else [] for role in PAYLOAD_NAMES}


def validate_published_attempt(attempt_root: Path) -> dict[str, Any]:
    completed = Path(attempt_root) / COMPLETED_NAME
    if not completed.is_dir() or not _validate_staging_files(completed, require_complete=True):
        raise AttemptStateError(f"completed attempt is not an exact complete set: {completed}")
    lock = strict_json_load(completed / "attempt.lock.json")
    paired = strict_json_load(completed / "paired_summary.json")
    release = strict_json_load(completed / "release_decision.json")
    manifest = strict_json_load(completed / "complete_set_manifest.json")
    if not all(isinstance(value, Mapping) for value in (lock, paired, release, manifest)):
        raise AttemptStateError("published JSON artifacts must be objects")
    if any(item.get("schema_version") != SCHEMA_VERSION for item in (lock, paired, release, manifest)):
        raise AttemptStateError("published artifacts have incompatible schema versions")
    identity = validate_identity(lock.get("identity"))
    if any(item.get("identity") != identity for item in (paired, release, manifest)):
        raise IdentityMismatchError("published artifacts do not share the attempt identity")
    expected_identity_sha = identity_sha256(identity)
    if any(item.get("identity_sha256") != expected_identity_sha for item in (lock, paired, release, manifest)):
        raise IdentityMismatchError("published identity digest mismatch")
    source_paths = lock.get("source_paths")
    if not isinstance(source_paths, Mapping):
        raise IdentityMismatchError("published lock is missing explicit source paths")
    source_paths = {str(key): str(value) for key, value in source_paths.items()}
    _validate_source_bindings(identity, source_paths)
    if any(item.get("source_paths") != source_paths for item in (paired, release, manifest)):
        raise IdentityMismatchError("published source bindings disagree with the attempt lock")
    if lock.get("status") != "completed" or manifest.get("complete_set") is not True:
        raise AttemptStateError("published attempt is not closed")
    if paired.get("status") != "evaluated":
        raise AttemptStateError("published paired summary is not evaluated")
    expected_release_status = "release_authorized" if release.get("release") is True else "release_rejected"
    if release.get("status") != expected_release_status:
        raise AttemptStateError("release decision status disagrees with release value")
    cases = _normalise_cases(paired.get("cases", []))
    input_identity_valid = identity["input_identity_sha256"] == input_identity_for_cases(cases)
    if not input_identity_valid:
        raise IdentityMismatchError("published input identity does not match published cases")
    expected_hashes = lock.get("payload_hashes")
    if not isinstance(expected_hashes, Mapping) or set(expected_hashes) != set(PAYLOAD_NAMES):
        raise AttemptStateError("lock is missing the exact two payload hashes")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list) or len(payloads) != 2 or sorted(item.get("name") for item in payloads if isinstance(item, Mapping)) != sorted(PAYLOAD_NAMES):
        raise AttemptStateError("complete-set manifest must contain baseline and candidate exactly once")
    for item in payloads:
        if not isinstance(item, Mapping) or item.get("name") not in PAYLOAD_NAMES:
            raise AttemptStateError("invalid payload manifest entry")
        name = str(item["name"])
        path = completed / str(item.get("path", ""))
        if path != completed / "payloads" / f"{name}.json" or not path.is_file():
            raise AttemptStateError(f"payload path mismatch: {name}")
        actual = sha256_file(path)
        if actual != expected_hashes.get(name) or actual != item.get("sha256") or item.get("expected_sha256") != expected_hashes.get(name) or item.get("bytes") != path.stat().st_size:
            raise AttemptStateError(f"payload hash mismatch: {name}")
    worker_checks = {role: _worker_check(completed / "payloads" / f"{role}.json", role, identity, cases) for role in PAYLOAD_NAMES}
    if manifest.get("worker_checks") != worker_checks or paired.get("worker_checks") != worker_checks:
        raise AttemptStateError("published worker checks are not derived from the payloads")
    comparison = paired.get("comparison_result")
    if not isinstance(comparison, Mapping) or paired.get("comparison_result_sha256") != sha256_bytes(_canonical_json(comparison)):
        raise AttemptStateError("comparison result is not bound to the published bytes")
    comparison = dict(comparison)
    comparison_valid = _comparison_check(comparison, identity, cases)
    trace_rows = _artifact_rows_from_summary(paired, "traces")
    video_rows = _artifact_rows_from_summary(paired, "videos")
    trace_results, trace_gate = _validate_artifact_bundle(trace_rows, cases, validate_trace)
    video_results, video_gate = _validate_artifact_bundle(video_rows, cases, validate_mp4)
    artifacts = _artifact_summary(trace_results, video_results)
    if artifacts != paired.get("artifacts") or artifacts != manifest.get("artifact_checks"):
        raise AttemptStateError("published artifact checks are not reproducible from their real files")
    if manifest.get("comparison_result_sha256") != paired.get("comparison_result_sha256"):
        raise AttemptStateError("manifest comparison result link mismatch")
    source_valid = True
    gates = _derive_gates(
        comparison=comparison,
        comparison_valid=comparison_valid,
        worker_checks=worker_checks,
        trace_gate=trace_gate,
        video_gate=video_gate,
        source_valid=source_valid,
        input_identity_valid=input_identity_valid,
        complete_set_gate=True,
    )
    if paired.get("gates") != gates or release.get("gates") != gates or manifest.get("gates") != gates:
        raise AttemptStateError("paired, release and manifest gates disagree with derived gates")
    if paired.get("release") is not release.get("release") or release.get("release") is not manifest.get("release") or release.get("release") is not all(gates.values()):
        raise AttemptStateError("release values disagree with derived gates")
    paired_sha = sha256_file(completed / "paired_summary.json")
    release_sha = sha256_file(completed / "release_decision.json")
    if release.get("paired_summary_sha256") != paired_sha or manifest.get("paired_summary_sha256") != paired_sha:
        raise AttemptStateError("release-to-paired summary hash mismatch")
    if release.get("comparison_result_sha256") != paired.get("comparison_result_sha256"):
        raise AttemptStateError("release-to-paired comparison hash mismatch")
    if manifest.get("release_decision_sha256") != release_sha:
        raise AttemptStateError("manifest-to-release hash mismatch")
    payload_hashes = {name: expected_hashes[name] for name in PAYLOAD_NAMES}
    request_fp = _request_fingerprint(identity, source_paths, cases, payload_hashes, comparison, artifacts)
    if any(item.get("request_fingerprint") != request_fp for item in (lock, paired, release, manifest)):
        raise IdentityMismatchError("completed request fingerprint cannot be recomputed from published content")
    return {"status": "completed", "release": bool(release["release"]), "identity": identity, "gates": gates, "manifest": manifest}


class FinalizerV2:
    """Close one immutable attempt, or resume its exact staging directory."""

    def __init__(self, attempt_root: Path, expected_identity: Mapping[str, Any], source_paths: Mapping[str, Path | str]):
        self.attempt_root = Path(attempt_root)
        self.expected_identity = validate_identity(expected_identity)
        if not isinstance(source_paths, Mapping) or set(source_paths) != set(SOURCE_IDENTITY_FIELDS):
            raise IdentityMismatchError("source_paths must name comparator, adapter and both plans")
        self.source_paths = {field: str(Path(source_paths[field]).resolve()) for field in SOURCE_IDENTITY_FIELDS}

    @property
    def staging(self) -> Path:
        return self.attempt_root / STAGING_NAME

    @property
    def completed(self) -> Path:
        return self.attempt_root / COMPLETED_NAME

    def _open_attempt(self, request_fingerprint: str) -> dict[str, Any]:
        if self.completed.exists():
            raise FileExistsError(f"completed attempt is immutable: {self.completed}")
        self.attempt_root.mkdir(parents=True, exist_ok=True)
        unexpected_root = [item for item in self.attempt_root.iterdir() if item.name not in {STAGING_NAME, COMPLETED_NAME}]
        if unexpected_root:
            raise AttemptStateError(f"unexpected attempt-root artifact: {unexpected_root[0]}")
        if self.staging.exists():
            lock = strict_json_load(self.staging / "attempt.lock.json")
            if lock.get("identity") != self.expected_identity or lock.get("source_paths") != self.source_paths:
                raise IdentityMismatchError("recovery identity or source binding differs from the staged attempt")
            if lock.get("request_fingerprint") != request_fingerprint:
                raise IdentityMismatchError("recovery input identity differs from the staged attempt")
            if lock.get("status") not in {"staging", "ready_to_publish", "completed"}:
                raise AttemptStateError("staged attempt has an unknown status")
            return lock
        self.staging.mkdir(parents=True)
        lock = {
            "schema_version": SCHEMA_VERSION,
            "status": "staging",
            "identity": self.expected_identity,
            "identity_sha256": identity_sha256(self.expected_identity),
            "source_paths": self.source_paths,
            "request_fingerprint": request_fingerprint,
            "payload_hashes": {},
            "payloads_written": [],
        }
        _atomic_json(self.staging / "attempt.lock.json", lock)
        _fsync_dir(self.attempt_root)
        return lock

    def finalize(
        self,
        payloads: Mapping[str, bytes | bytearray | Mapping[str, Any] | Path | str],
        *,
        cases: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        traces: Mapping[str, Any] | None,
        videos: Mapping[str, Any] | None,
        comparison_result: Mapping[str, Any] | Path | str | None,
        identity: Mapping[str, Any] | None = None,
        summary_metadata: Mapping[str, Any] | None = None,
        crash_after: str | None = None,
    ) -> dict[str, Any]:
        requested_identity = validate_identity(identity or self.expected_identity)
        if requested_identity != self.expected_identity:
            raise IdentityMismatchError("requested identity does not match the finalizer identity")
        _validate_source_bindings(requested_identity, self.source_paths)
        case_map = _normalise_cases(cases)
        input_identity_valid = requested_identity["input_identity_sha256"] == input_identity_for_cases(case_map)
        if not isinstance(payloads, Mapping) or set(payloads) != set(PAYLOAD_NAMES):
            raise AttemptStateError("exactly baseline and candidate worker payloads are required")
        payload_data = {name: _payload_bytes(payloads[name]) for name in PAYLOAD_NAMES}
        payload_hashes = {name: sha256_bytes(data) for name, data in payload_data.items()}
        comparison = _read_comparison(comparison_result)
        comparison_valid = _comparison_check(comparison, requested_identity, case_map)
        trace_results, trace_gate = _validate_artifact_bundle(traces, case_map, validate_trace)
        video_results, video_gate = _validate_artifact_bundle(videos, case_map, validate_mp4)
        artifacts = _artifact_summary(trace_results, video_results)
        request_fp = _request_fingerprint(requested_identity, self.source_paths, case_map, payload_hashes, comparison, artifacts)
        lock = self._open_attempt(request_fp)
        if not _validate_staging_files(self.staging, require_complete=False):
            raise AttemptStateError("staging directory contains an unexpected or partial artifact")
        lock_payload_hashes = lock.get("payload_hashes")
        if lock_payload_hashes and lock_payload_hashes != payload_hashes:
            raise IdentityMismatchError("recovery payload identity differs from the staged attempt")
        lock["payload_hashes"] = payload_hashes
        for name in PAYLOAD_NAMES:
            path = self.staging / "payloads" / f"{name}.json"
            existed = path.exists()
            _write_once(path, payload_data[name])
            if not existed:
                lock["payloads_written"] = sorted(set(lock.get("payloads_written", [])) | {name})
                _atomic_json(self.staging / "attempt.lock.json", lock)
            if name == "baseline" and crash_after == "first_payload" and not existed:
                raise InjectedCrash("injected interruption after first payload")
        if not _validate_staging_files(self.staging, require_complete=False) or not {"payloads/baseline.json", "payloads/candidate.json"} <= _file_set(self.staging):
            raise AttemptStateError("staging directory is not an exact two-payload set")
        worker_checks = {role: _worker_check(self.staging / "payloads" / f"{role}.json", role, requested_identity, case_map) for role in PAYLOAD_NAMES}
        gates = _derive_gates(
            comparison=comparison,
            comparison_valid=comparison_valid,
            worker_checks=worker_checks,
            trace_gate=trace_gate,
            video_gate=video_gate,
            source_valid=True,
            input_identity_valid=input_identity_valid,
            complete_set_gate=True,
        )
        release = bool(all(gates.values()))
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "evaluated",
            "identity": requested_identity,
            "identity_sha256": identity_sha256(requested_identity),
            "source_paths": self.source_paths,
            "request_fingerprint": request_fp,
            "cases": [{"case_id": case_id, "case_fingerprint": case_map[case_id]} for case_id in sorted(case_map)],
            "payload_names": list(PAYLOAD_NAMES),
            "worker_checks": worker_checks,
            "comparison_result": comparison,
            "comparison_result_sha256": sha256_bytes(_canonical_json(comparison)),
            "artifacts": artifacts,
            "gates": gates,
            "release": release,
            "metadata": dict(summary_metadata or {}),
        }
        summary_path = self.staging / "paired_summary.json"
        summary_existed = summary_path.exists()
        _write_once(summary_path, _canonical_json(summary) + b"\n")
        if crash_after == "paired_summary" and not summary_existed:
            raise InjectedCrash("injected interruption after paired summary")
        release_decision = {
            "schema_version": SCHEMA_VERSION,
            "status": "release_authorized" if release else "release_rejected",
            "release": release,
            "gates": gates,
            "identity": requested_identity,
            "identity_sha256": identity_sha256(requested_identity),
            "source_paths": self.source_paths,
            "request_fingerprint": request_fp,
            "paired_summary_sha256": sha256_file(summary_path),
            "comparison_result_sha256": sha256_bytes(_canonical_json(comparison)),
            "fold_authorized": False,
            "zip_authorized": False,
            "auto_upload": False,
            "auto_submit": False,
        }
        release_path = self.staging / "release_decision.json"
        release_existed = release_path.exists()
        _write_once(release_path, _canonical_json(release_decision) + b"\n")
        if crash_after == "release_decision" and not release_existed:
            raise InjectedCrash("injected interruption after release decision")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "complete_set": True,
            "identity": requested_identity,
            "identity_sha256": identity_sha256(requested_identity),
            "source_paths": self.source_paths,
            "request_fingerprint": request_fp,
            "payloads": _payload_descriptors(self.staging, payload_hashes),
            "worker_checks": worker_checks,
            "comparison_result_sha256": sha256_bytes(_canonical_json(comparison)),
            "artifact_checks": artifacts,
            "gates": gates,
            "release": release,
            "paired_summary_sha256": sha256_file(summary_path),
            "release_decision_sha256": sha256_file(release_path),
        }
        manifest_path = self.staging / "complete_set_manifest.json"
        manifest_existed = manifest_path.exists()
        _write_once(manifest_path, _canonical_json(manifest) + b"\n")
        if crash_after == "manifest" and not manifest_existed:
            raise InjectedCrash("injected interruption after complete-set manifest")
        if not _validate_staging_files(self.staging, require_complete=True):
            raise AttemptStateError("complete-set manifest does not close an exact artifact set")
        lock.update({"status": "completed", "payloads_written": list(PAYLOAD_NAMES)})
        _atomic_json(self.staging / "attempt.lock.json", lock)
        if self.completed.exists():
            raise FileExistsError(f"completed attempt is immutable: {self.completed}")
        os.replace(self.staging, self.completed)
        _fsync_dir(self.attempt_root)
        if crash_after == "rename":
            raise InjectedCrash("injected interruption after directory rename")
        return validate_published_attempt(self.attempt_root)


__all__ = [
    "AttemptStateError",
    "COMPARISON_RESULT_SCHEMA",
    "COMPLETED_NAME",
    "FinalizerV2",
    "FinalizerV2Error",
    "IDENTITY_FIELDS",
    "InjectedCrash",
    "IdentityMismatchError",
    "PAYLOAD_NAMES",
    "SCHEMA_VERSION",
    "SOURCE_IDENTITY_FIELDS",
    "STAGING_NAME",
    "WORKER_RESULT_SCHEMA",
    "ZERO_TRACE_POLICY",
    "identity_sha256",
    "input_identity_for_cases",
    "sha256_file",
    "strict_json_load",
    "validate_identity",
    "validate_mp4",
    "validate_published_attempt",
    "validate_trace",
]
