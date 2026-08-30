"""Serve an OpenPI Axis Pi0.5 checkpoint through the local evaluator HTTP API.

The repository's evaluator intentionally uses the small HTTP/msgpack protocol in
``pipeline.remote_policy``.  OpenPI's reference server uses a websocket protocol
and returns an action chunk, so this adapter is the narrow compatibility layer:
it preserves the OpenPI LIBERO preprocessing contract and consumes one action
from a chunk per evaluator request.

This module keeps the OpenPI import lazy so its observation conversion helpers
remain testable in the lightweight evaluator environment.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import msgpack
import numpy as np

LOGGER = logging.getLogger(__name__)
ACTION_DIM = 7
ACTION_HORIZON = 10
MODEL_IMAGE_SIZE = 224


def deserialize_observation(payload: bytes) -> dict[str, np.ndarray]:
    """Decode the evaluator's explicit-array msgpack observation envelope."""

    unpacked = msgpack.unpackb(payload, raw=False)
    observation: dict[str, np.ndarray] = {}
    for key, value in unpacked.items():
        if not isinstance(value, dict):
            raise ValueError(f"observation field {key!r} is not an array envelope")
        try:
            array = np.frombuffer(
                value["data"], dtype=np.dtype(value["dtype"])
            ).reshape(value["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid observation field {key!r}") from exc
        observation[str(key)] = array.copy()
    return observation


def serialize_action(action: np.ndarray) -> bytes:
    """Encode one float32 seven-dimensional action for the evaluator client."""

    array = np.asarray(action, dtype=np.float32)
    if array.shape != (ACTION_DIM,):
        raise ValueError(f"expected action shape {(ACTION_DIM,)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("action contains NaN or Inf")
    return msgpack.packb({"data": array.tobytes()}, use_bin_type=True)


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert robosuite's xyzw quaternion representation to axis-angle."""

    value = np.asarray(quat, dtype=np.float32).reshape(-1).copy()
    if value.shape != (4,):
        raise ValueError(f"expected quaternion shape (4,), got {value.shape}")
    scalar = float(np.clip(value[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(0.0, 1.0 - scalar * scalar)))
    if np.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (value[:3] * (2.0 * np.arccos(scalar)) / denominator).astype(np.float32)


def _resize_for_openpi(image: np.ndarray) -> np.ndarray:
    """Rotate and resize a LIBERO camera image as in OpenPI's LIBERO example."""

    from openpi_client import image_tools

    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {value.shape}")
    # LIBERO's renderer is vertically inverted relative to the training frames.
    value = np.ascontiguousarray(value[::-1, ::-1])
    return np.ascontiguousarray(
        image_tools.convert_to_uint8(
            image_tools.resize_with_pad(value, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
        )
    )


def build_openpi_observation(
    observation: dict[str, np.ndarray], instruction: str
) -> dict[str, Any]:
    """Map current evaluator observations to the v3 OpenPI LIBERO contract."""

    required = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    missing = [key for key in required if key not in observation]
    if missing:
        raise KeyError(f"missing evaluator observation keys: {missing}")

    state = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(3),
            quat_to_axisangle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
        )
    ).astype(np.float32)
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise ValueError(f"invalid LIBERO state: shape={state.shape}")

    # ``create_trained_policy`` applies model/data transforms at inference time,
    # but dataset repacking is a loader concern and is not applied there.  The
    # LIBERO policy transform therefore consumes the canonical inference names.
    return {
        "observation/image": _resize_for_openpi(
            observation["agentview_image"]
        ),
        "observation/wrist_image": _resize_for_openpi(
            observation["robot0_eye_in_hand_image"]
        ),
        "observation/state": state,
        "prompt": str(instruction),
    }


class CheckpointPolicy:
    """OpenPI policy plus the action-chunk state expected by the HTTP client."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        config_name: str = "axis_pi05_libero_mix",
        default_prompt: str | None = None,
    ) -> None:
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {checkpoint}")
        # These imports initialize JAX/model state and are intentionally startup-only.
        from openpi.policies import policy_config
        from openpi.training import config as training_config
        from openpi.shared import normalize
        import jax

        self._jax = jax
        config = training_config.get_config(config_name)
        norm_path = os.environ.get("AXIS_PI05_NORM_STATS")
        norm_stats = normalize.deserialize_json(Path(norm_path).read_text(encoding="utf-8")) if norm_path else None
        self._policy = policy_config.create_trained_policy(
            config,
            checkpoint,
            default_prompt=default_prompt,
            norm_stats=norm_stats,
        )
        self._actions: collections.deque[np.ndarray] = collections.deque()
        self._instruction = default_prompt or ""
        self._policy_seed: int | None = None
        self._lock = threading.Lock()
        LOGGER.info("loaded checkpoint %s with config %s", checkpoint, config_name)

    def reset(
        self,
        instruction: str = "",
        seed: int | None = None,
        *,
        reset_kind: str = "episode_reset",
        chunk_index: int = 0,
    ) -> None:
        if reset_kind not in {"episode_reset", "chunk_discard"}:
            raise ValueError(f"unknown policy reset kind: {reset_kind}")
        if reset_kind == "episode_reset" and seed is None:
            raise ValueError("episode_reset requires an integer policy_seed")
        if reset_kind == "chunk_discard" and seed is not None:
            raise ValueError("chunk_discard must not reseed the policy")
        with self._lock:
            self._actions.clear()
            self._instruction = instruction
            if reset_kind == "episode_reset":
                # The evaluator supplies a frozen policy_seed for every
                # episode.  Reset the policy object and JAX sampling stream
                # only at the episode boundary.
                self._policy.reset()
                self._policy._rng = self._jax.random.key(int(seed))  # noqa: SLF001
                self._policy_seed = int(seed)
            # A chunk discard intentionally only clears the unexecuted action
            # buffer.  It must not call Policy.reset() or reseed JAX, so the
            # next inference continues the same episode RNG stream.
            LOGGER.info(
                "policy reset reset_kind=%s episode_seed=%s chunk_index=%d reseed=%s instruction=%r",
                reset_kind,
                self._policy_seed,
                int(chunk_index),
                reset_kind == "episode_reset",
                instruction,
            )

    def get_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        with self._lock:
            if not self._actions:
                model_observation = build_openpi_observation(
                    observation, self._instruction
                )
                result = self._policy.infer(model_observation)
                actions = np.asarray(result["actions"], dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions[None, :]
                if actions.ndim != 2 or actions.shape[1] < ACTION_DIM:
                    raise ValueError(f"invalid OpenPI action chunk shape: {actions.shape}")
                if not np.all(np.isfinite(actions)):
                    raise ValueError("OpenPI action chunk contains NaN or Inf")
                self._actions.extend(actions[:, :ACTION_DIM].copy())
            return self._actions.popleft()


def make_handler(policy: CheckpointPolicy) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AxisPi05CheckpointHTTP/1"

        def log_message(self, format: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP hook
            if self.path == "/health":
                self._send(200, "application/json", b'{"status":"ok"}')
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found\n")

        def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP hook
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length)
            try:
                if self.path == "/reset":
                    body = json.loads(payload.decode("utf-8")) if payload else {}
                    policy.reset(
                        instruction=str(body.get("instruction", "")),
                        seed=body.get("seed"),
                        reset_kind=str(body.get("reset_kind", "episode_reset")),
                        chunk_index=int(body.get("chunk_index", 0)),
                    )
                    self._send(200, "application/json", b'{"status":"ok"}')
                    return
                if self.path == "/act":
                    action = policy.get_action(deserialize_observation(payload))
                    self._send(200, "application/x-msgpack", serialize_action(action))
                    return
                self._send(404, "text/plain; charset=utf-8", b"not found\n")
            except Exception as exc:  # return a diagnosable protocol error
                LOGGER.exception("request failed")
                self._send(500, "text/plain; charset=utf-8", f"{exc}\n".encode())

    return Handler


def serve(
    checkpoint: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    config_name: str = "axis_pi05_libero_mix",
    default_prompt: str | None = None,
) -> None:
    policy = CheckpointPolicy(
        checkpoint, config_name=config_name, default_prompt=default_prompt
    )
    server = ThreadingHTTPServer((host, port), make_handler(policy))
    LOGGER.info("serving %s:%d", host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="axis_pi05_libero_mix")
    parser.add_argument("--default-prompt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    serve(
        args.checkpoint,
        host=args.host,
        port=args.port,
        config_name=args.config,
        default_prompt=args.default_prompt,
    )


if __name__ == "__main__":
    main()
