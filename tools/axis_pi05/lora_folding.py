"""Pure NumPy helpers for materializing the Axis LoRA parameter tree."""

from __future__ import annotations

from typing import Any

import numpy as np


def fold_last_two_axes(base: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return base + left@right for a batched matrix-shaped JAX weight."""

    if base.ndim < 2 or left.ndim != base.ndim or right.ndim != base.ndim:
        raise ValueError(
            f"unsupported LoRA shapes: base={base.shape} left={left.shape} right={right.shape}"
        )
    if left.shape[:-1] != base.shape[:-1] or right.shape[:-2] != base.shape[:-2]:
        raise ValueError(
            f"LoRA leading dimensions do not match: base={base.shape} left={left.shape} right={right.shape}"
        )
    if left.shape[-2] != base.shape[-2] or right.shape[-1] != base.shape[-1]:
        raise ValueError(
            f"LoRA output dimensions do not match: base={base.shape} left={left.shape} right={right.shape}"
        )
    if left.shape[-1] != right.shape[-2]:
        raise ValueError(f"LoRA rank mismatch: left={left.shape} right={right.shape}")
    delta = np.einsum("...ir,...rj->...ij", left, right, optimize=True)
    return np.asarray(base, dtype=np.float32) + np.asarray(delta, dtype=np.float32)


def fold_lora_in_mapping(node: Any) -> int:
    """Recursively fold all Axis Flax LoRA leaves in-place and return a count."""

    if not isinstance(node, dict):
        return 0

    folded = 0
    if all(key in node for key in ("w", "lora_a", "lora_b")):
        node["w"] = fold_last_two_axes(node["w"], node["lora_a"], node["lora_b"])
        del node["lora_a"], node["lora_b"]
        folded += 1

    for base_name in ("gating_einsum", "linear"):
        left_name = f"{base_name}_lora_a"
        right_name = f"{base_name}_lora_b"
        if base_name in node and left_name in node and right_name in node:
            node[base_name] = fold_last_two_axes(node[base_name], node[left_name], node[right_name])
            del node[left_name], node[right_name]
            folded += 1

    for value in node.values():
        folded += fold_lora_in_mapping(value)
    return folded


def assert_no_lora(node: Any, path: str = "params") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if "lora" in str(key).lower():
                raise ValueError(f"unfolded LoRA parameter at {path}/{key}")
            assert_no_lora(value, f"{path}/{key}")
