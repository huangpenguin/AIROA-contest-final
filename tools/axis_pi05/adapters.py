from __future__ import annotations

from typing import Any, TypedDict

from .contract import validate_frame


CanonicalFrame = TypedDict("CanonicalFrame", {
    "observation.images.front": Any,
    "observation.images.wrist": Any,
    "observation.state": list[float],
    "action": list[float],
    "timestamp": float,
    "source_fps": int,
    "source": str,
    "source_repo": str,
    "revision": str,
    "source_episode_index": int,
    "suite": str,
    "task": str,
    "task_index": int,
    "instruction": str,
    "perturb_category": str,
    "perturb_label_quality": str,
})


def canonicalize(raw: dict[str, Any], episode: dict[str, Any]) -> CanonicalFrame:
    """Rename keys/namespaces only; never resample image/action/state streams."""
    front = raw.get("observation.images.front", raw.get("observation.images.image"))
    wrist = raw.get("observation.images.wrist", raw.get("observation.images.wrist_image"))
    state = raw.get("observation.state")
    action = raw.get("action", raw.get("actions"))
    result: CanonicalFrame = {
        "observation.images.front": front,
        "observation.images.wrist": wrist,
        "observation.state": [float(value) for value in state],
        "action": [float(value) for value in action],
        "timestamp": float(raw.get("timestamp", 0.0)),
        "source_fps": int(episode["source_fps"]),
        "source": str(episode["source"]),
        "source_repo": str(episode["source_repo"]),
        "revision": str(episode["revision"]),
        "source_episode_index": int(episode["source_episode_index"]),
        "suite": str(episode["suite"]),
        "task": str(episode["task"]),
        "task_index": int(episode["task_index"]),
        "instruction": str(episode["instruction"]),
        "perturb_category": str(episode["perturb_category"]),
        "perturb_label_quality": str(episode["perturb_label_quality"]),
    }
    errors = validate_frame(result)
    if errors:
        raise ValueError("canonical frame contract failed: " + "; ".join(errors))
    return result
