"""Auditable Pi0.5-Axis post-training tools.

The package intentionally keeps dataset/model media outside Git.  Commands in
this package operate on tracked locks and manifests, and fail closed when a
revision, schema field, or dependency cannot be verified.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
