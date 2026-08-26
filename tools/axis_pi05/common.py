from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
SHA256_RE = r"^[0-9a-f]{64}$"
FULL_SHA_RE = r"^[0-9a-f]{40}$"


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - bootstrap catches this
        raise RuntimeError("需要 PyYAML；请在训练环境中安装 requirements/uv lock") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"无法读取 YAML: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"YAML 顶层必须是 mapping: {path}")
    return value


def dump_yaml(value: Any) -> str:
    import yaml

    return yaml.safe_dump(value, allow_unicode=True, sort_keys=False)


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    return path if path.is_absolute() else base / path


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path, *, exclude: Iterable[str] = ()) -> str:
    excluded = set(exclude)
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in excluded or any(part in excluded for part in path.parts):
            continue
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: 非法 JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row 必须是 object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_table(path: Path, *, columns: list[str] | None = None) -> list[dict[str, Any]]:
    """Read a Parquet table without forcing pandas into the lightweight CLI.

    Actual data execution requires either pyarrow or pandas+fastparquet.  This
    explicit requirement is preferable to silently reading only metadata or
    changing the frame stream.
    """
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"读取 {path} 需要 pyarrow 或 pandas+fastparquet；不会用替代格式伪装 Parquet"
            ) from exc
        frame = pd.read_parquet(path, columns=columns)
        return frame.to_dict(orient="records")
    table = pq.read_table(path, columns=columns)
    return table.to_pylist()


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the tracked manifest format as Parquet; fail instead of faking it."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"写入 {path} 需要 pyarrow 或 pandas+fastparquet；不会把 JSONL 冒充 Parquet"
            ) from exc
        pd.DataFrame(rows).to_parquet(path, index=False)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def find_single(path: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        candidates = sorted(path.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def git_info(root: Path = ROOT) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()

    status = git("status", "--porcelain")
    diff = subprocess.run(["git", "-C", str(root), "diff", "--binary"], capture_output=True, check=True).stdout
    cached = subprocess.run(["git", "-C", str(root), "diff", "--cached", "--binary"], capture_output=True, check=True).stdout
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "diff_sha256": sha256_bytes(diff + cached),
    }


def iter_files(root: Path) -> Iterator[Path]:
    yield from (path for path in sorted(root.rglob("*")) if path.is_file())
