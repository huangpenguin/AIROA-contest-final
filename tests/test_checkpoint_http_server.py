import collections
import threading

import msgpack
import numpy as np

from tools.axis_pi05.checkpoint_http_server import (
    CheckpointPolicy,
    build_openpi_observation,
    deserialize_observation,
    serialize_action,
)


def test_checkpoint_policy_reset_uses_episode_policy_seed():
    class FakeRandom:
        @staticmethod
        def key(seed):
            return ("jax-key", int(seed))

    class FakeJax:
        random = FakeRandom()

    class FakePolicy:
        def __init__(self):
            self.reset_count = 0
            self._rng = None

        def reset(self):
            self.reset_count += 1

    policy = CheckpointPolicy.__new__(CheckpointPolicy)
    policy._jax = FakeJax()
    policy._policy = FakePolicy()
    policy._actions = collections.deque([np.ones(7, dtype=np.float32)])
    policy._instruction = "old"
    policy._policy_seed = None
    policy._lock = threading.Lock()

    policy.reset(instruction="new", seed=54000)

    assert policy._actions == collections.deque()
    assert policy._instruction == "new"
    assert policy._policy.reset_count == 1
    assert policy._policy._rng == ("jax-key", 54000)
    assert policy._policy_seed == 54000

    policy.reset(instruction="new", reset_kind="chunk_discard", chunk_index=1)

    assert policy._policy.reset_count == 1
    assert policy._policy._rng == ("jax-key", 54000)
    assert policy._policy_seed == 54000


def _observation_envelope(observation):
    return msgpack.packb(
        {
            key: {
                "data": value.tobytes(),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in observation.items()
        },
        use_bin_type=True,
    )


def test_observation_roundtrip_and_openpi_mapping():
    source = {
        "agentview_image": np.zeros((4, 5, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((4, 5, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([1, 2, 3], dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.array([4, 5], dtype=np.float32),
    }
    decoded = deserialize_observation(_observation_envelope(source))
    mapped = build_openpi_observation(decoded, "pick up the block")
    assert mapped["observation/image"].shape == (224, 224, 3)
    assert mapped["observation/wrist_image"].shape == (224, 224, 3)
    np.testing.assert_allclose(mapped["observation/state"], [1, 2, 3, 0, 0, 0, 4, 5])
    assert mapped["prompt"] == "pick up the block"


def test_action_serialization_enforces_contract():
    packed = serialize_action(np.arange(7, dtype=np.float64))
    payload = msgpack.unpackb(packed, raw=False)
    np.testing.assert_array_equal(
        np.frombuffer(payload["data"], dtype=np.float32), np.arange(7, dtype=np.float32)
    )
