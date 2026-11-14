"""Validated live RTP settings and a small, safe YAML boundary."""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RtpConfig:
    port: int = 5600
    payload_type: int = 96
    clock_rate: int = 90000
    encoding: str = "H264"
    udp_buffer_size_bytes: int = 2097152
    decoder: str = "vtdec_hw"
    expected_width_px: int = 1920
    expected_height_px: int = 1440
    expected_fps: float = 30.0
    timeout_s: float = 10.0
    max_frames: int | None = None

    def __post_init__(self) -> None:
        for name, low, high in (
            ("port", 1, 65535),
            ("payload_type", 0, 127),
            ("clock_rate", 90000, 90000),
            ("udp_buffer_size_bytes", 1, 2**31 - 1),
            ("expected_width_px", 1, 8192),
            ("expected_height_px", 1, 8192),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.encoding != "H264":
            raise ValueError("encoding must be H264")
        if not isinstance(self.decoder, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]{0,63}", self.decoder
        ):
            raise ValueError("decoder must be a single GStreamer element name")
        for name, high in (("expected_fps", 1000), ("timeout_s", 86400)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not 0 < value <= high
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and in (0, {high}]")
        if self.max_frames is not None and (
            type(self.max_frames) is not int or self.max_frames <= 0
        ):
            raise ValueError("max_frames must be a positive integer")


class _UniqueLoader(yaml.SafeLoader):
    """Reject ambiguous duplicate keys, aliases, and YAML object constructors."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are not supported")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> Any:
        seen: set[str] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in seen:
                raise ValueError("YAML keys must be unique strings")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _section(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    unknown = value.keys() - keys
    if unknown:
        raise ValueError(f"unknown {name} keys: {', '.join(sorted(unknown))}")
    return value


def load_rtp_config(path: Path) -> RtpConfig:
    """Load at most 64 KiB of settings; omitted fields use RtpConfig defaults."""
    try:
        if not path.is_file():
            raise ValueError(
                f"cannot read live configuration: not a regular file: {path}"
            )
        with path.open("rb") as handle:
            data = handle.read(65537)
        if len(data) > 65536:
            raise ValueError("live YAML configuration exceeds 64 KiB")
        root = _section(yaml.load(data, Loader=_UniqueLoader), {"camera"}, "root")
        camera = _section(
            root.get("camera"),
            {"source", "rtp", "decoder", "expected", "timeout_s", "max_frames"},
            "camera",
        )
        if camera.get("source") != "rtp":
            raise ValueError("camera.source must be rtp")
        rtp = _section(
            camera.get("rtp", {}),
            {"port", "payload_type", "clock_rate", "encoding", "udp_buffer_size_bytes"},
            "camera.rtp",
        )
        expected = _section(
            camera.get("expected", {}),
            {"width_px", "height_px", "fps"},
            "camera.expected",
        )
        values = dict(rtp)
        for key in ("decoder", "timeout_s", "max_frames"):
            if key in camera:
                values[key] = camera[key]
        values.update({f"expected_{key}": value for key, value in expected.items()})
        return RtpConfig(**values)
    except (OSError, yaml.YAMLError, RecursionError) as exc:
        raise ValueError(f"cannot read live configuration {path}: {exc}") from exc
