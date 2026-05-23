import pytest
import torch

from babappalign import devices


def test_device_choices_include_mps_and_auto():
    assert devices.DEVICE_CHOICES == ("auto", "cpu", "cuda", "mps")


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(devices, "cuda_is_available", lambda: True)
    monkeypatch.setattr(devices, "mps_is_available", lambda: True)
    monkeypatch.setattr(devices, "device_is_usable", lambda device: True)

    assert devices.resolve_device("auto") == torch.device("cuda")


def test_auto_uses_mps_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(devices, "cuda_is_available", lambda: False)
    monkeypatch.setattr(devices, "mps_is_available", lambda: True)
    monkeypatch.setattr(devices, "device_is_usable", lambda device: True)

    assert devices.resolve_device("auto") == torch.device("mps")


def test_auto_skips_accelerator_that_fails_runtime_probe(monkeypatch):
    monkeypatch.setattr(devices, "cuda_is_available", lambda: True)
    monkeypatch.setattr(devices, "mps_is_available", lambda: True)
    monkeypatch.setattr(devices, "device_is_usable", lambda device: False)

    assert devices.resolve_device("auto") == torch.device("cpu")


def test_unavailable_requested_accelerator_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(devices, "cuda_is_available", lambda: False)
    monkeypatch.setattr(devices, "mps_is_available", lambda: False)

    with pytest.warns(RuntimeWarning):
        assert devices.resolve_device("mps") == torch.device("cpu")
