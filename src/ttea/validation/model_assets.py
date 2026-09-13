"""Explicitly allow-listed validation assets; never a generic model downloader."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ttea.benchmarking.ollama import OllamaClient, OllamaRuntimeError


TINY_MODEL_NAME = "smollm2:135m-instruct-q4_0"


@dataclass(frozen=True)
class TinyModelManifest:
    model: str
    purpose: str
    download_timeout_seconds: float
    expected_disk_bytes: int
    ram_estimate_bytes: int
    gpu_vram_estimate_bytes: int

    def __post_init__(self) -> None:
        if self.model != TINY_MODEL_NAME:
            raise ValueError("tiny validation manifest must use the exact allow-listed model")
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise ValueError("purpose must be non-empty")
        if not isinstance(self.download_timeout_seconds, (int, float)) or isinstance(self.download_timeout_seconds, bool) or not math.isfinite(self.download_timeout_seconds) or self.download_timeout_seconds <= 0:
            raise ValueError("download_timeout_seconds must be finite and positive")
        for name in ("expected_disk_bytes", "ram_estimate_bytes", "gpu_vram_estimate_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


MIB = 1024 * 1024
TINY_MODEL_MANIFEST = TinyModelManifest(
    model=TINY_MODEL_NAME,
    purpose="T/TEA safety validation only; not an intelligence benchmark candidate",
    download_timeout_seconds=20 * 60,
    expected_disk_bytes=512 * MIB,
    ram_estimate_bytes=512 * MIB,
    gpu_vram_estimate_bytes=256 * MIB,
)
VALIDATION_MODEL_ALLOWLIST = {TINY_MODEL_NAME: TINY_MODEL_MANIFEST}


class ModelProvisionBlocked(RuntimeError):
    """The asset is absent and the campaign did not authorize acquisition."""


@dataclass(frozen=True)
class TinyModelMetadata:
    exact_model_name: str
    digest: str
    size_bytes: int
    parameter_size: str | None = None
    quantization: str | None = None
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.exact_model_name != TINY_MODEL_NAME:
            raise ValueError("metadata model identity does not match the tiny validation model")
        if not isinstance(self.digest, str) or not self.digest.strip():
            raise ValueError("digest must be non-empty")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        for value, name in ((self.parameter_size, "parameter_size"), (self.quantization, "quantization")):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty when present")
        if not isinstance(self.capabilities, tuple) or any(not isinstance(item, str) or not item.strip() for item in self.capabilities):
            raise ValueError("capabilities must be a tuple of non-empty strings")

    def evidence(self) -> dict[str, Any]:
        return {"model": self.exact_model_name, "digest": self.digest, "size_bytes": self.size_bytes,
                "parameter_size": self.parameter_size, "quantization": self.quantization,
                "capabilities": list(self.capabilities)}


def validation_manifest(model: str) -> TinyModelManifest:
    try:
        return VALIDATION_MODEL_ALLOWLIST[model]
    except (KeyError, TypeError) as error:
        raise ValueError("model is not allow-listed for T/TEA validation") from error


class TinyModelProvisioner:
    def __init__(self, manifest: TinyModelManifest = TINY_MODEL_MANIFEST) -> None:
        self.manifest = validation_manifest(manifest.model)

    def _record(self, client: OllamaClient) -> dict[str, Any] | None:
        return next((item for item in client.installed_model_records()
                     if (item.get("name") or item.get("model")) == self.manifest.model), None)

    def inspect(self, client: OllamaClient) -> TinyModelMetadata | None:
        record = self._record(client)
        if record is None:
            return None
        digest, size = record.get("digest"), record.get("size")
        if not isinstance(digest, str) or not digest.strip() or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise OllamaRuntimeError("installed tiny-model record lacks a valid digest or size")
        details = record.get("details") if isinstance(record.get("details"), dict) else {}
        try:
            shown = client.show_model(self.manifest.model)
        except OllamaRuntimeError:
            shown = {}
        shown_details = shown.get("details") if isinstance(shown.get("details"), dict) else {}
        parameter_size = shown_details.get("parameter_size") or details.get("parameter_size")
        quantization = shown_details.get("quantization_level") or details.get("quantization_level")
        capabilities = shown.get("capabilities", ())
        if not isinstance(capabilities, list) or any(not isinstance(item, str) or not item.strip() for item in capabilities):
            capabilities = ()
        return TinyModelMetadata(self.manifest.model, digest, size,
                                 parameter_size if isinstance(parameter_size, str) and parameter_size.strip() else None,
                                 quantization if isinstance(quantization, str) and quantization.strip() else None,
                                 tuple(capabilities))

    def ensure(self, client: OllamaClient, allow_download: bool) -> tuple[TinyModelMetadata, bool]:
        if not isinstance(allow_download, bool):
            raise ValueError("allow_download must be a bool")
        metadata = self.inspect(client)
        if metadata is not None:
            return metadata, False
        if not allow_download:
            raise ModelProvisionBlocked("tiny validation model is missing and model download was not authorized")
        client.pull(self.manifest.model, self.manifest.download_timeout_seconds)
        metadata = self.inspect(client)
        if metadata is None:
            raise OllamaRuntimeError("tiny validation model is still absent after pull")
        return metadata, True
