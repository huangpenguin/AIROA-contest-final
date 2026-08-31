"""Local-only startup preload for the offline FAST audit subprocess.

The submission bundle carries the official-image torchvision and sentencepiece
packages.  The audit venv uses a different torch/Python build, so preload its
matching host extensions before policy_server.py inserts the staged vendor
directory.  The file is not copied into the submission staging directory.
"""

try:
    import torchvision  # noqa: F401
except Exception:
    pass

try:
    import sentencepiece  # noqa: F401
except Exception:
    pass
