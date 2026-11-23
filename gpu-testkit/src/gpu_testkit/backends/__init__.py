"""Hardware discovery backends and the selection policy."""

from __future__ import annotations

import os
import shutil

from gpu_testkit.backends.base import Backend, NullBackend
from gpu_testkit.backends.fake import FakeBackend
from gpu_testkit.backends.nvml import NvmlBackend, load_pynvml
from gpu_testkit.backends.smi import SmiBackend

FAKE_SPEC_ENV = "GPU_TESTKIT_FAKE_SPEC"
BACKEND_CHOICES = ("auto", "fake", "nvml", "smi", "none")

__all__ = [
    "BACKEND_CHOICES",
    "FAKE_SPEC_ENV",
    "Backend",
    "BackendUnavailable",
    "FakeBackend",
    "NullBackend",
    "NvmlBackend",
    "SmiBackend",
    "select_backend",
]


class BackendUnavailable(RuntimeError):
    """An explicitly requested backend cannot run on this host."""


def select_backend(choice: str = "auto", fake_spec: str | None = None) -> Backend:
    """Pick a backend.

    ``auto`` order: a fake spec (option or ``GPU_TESTKIT_FAKE_SPEC``) wins, so CI
    never touches real devices by accident; then NVML; then ``nvidia-smi``; then none.
    An explicit choice that cannot work raises instead of silently degrading.
    """
    spec = fake_spec or os.environ.get(FAKE_SPEC_ENV)
    if choice == "fake" or (choice == "auto" and spec):
        if not spec:
            raise BackendUnavailable(f"--gpu-backend=fake needs --gpu-fake-spec or {FAKE_SPEC_ENV}")
        return FakeBackend.from_source(spec)
    if choice in ("auto", "nvml"):
        pynvml = load_pynvml()
        if pynvml is not None:
            return NvmlBackend(pynvml)
        if choice == "nvml":
            raise BackendUnavailable("--gpu-backend=nvml but pynvml is missing or nvmlInit failed")
    if choice in ("auto", "smi"):
        if shutil.which("nvidia-smi"):
            return SmiBackend()
        if choice == "smi":
            raise BackendUnavailable("--gpu-backend=smi but nvidia-smi is not on PATH")
    if choice not in BACKEND_CHOICES:
        raise BackendUnavailable(f"unknown backend {choice!r}; choose from {BACKEND_CHOICES}")
    return NullBackend()
