from pathlib import Path

import pytest

from ad4pt.gstreamer import build_pipeline
from ad4pt.live_config import RtpConfig, load_rtp_config


@pytest.mark.parametrize(
    "key,value",
    [
        ("port", 0),
        ("port", 65536),
        ("port", True),
        ("port", "5600"),
        ("payload_type", -1),
        ("payload_type", 128),
        ("payload_type", 1.5),
        ("clock_rate", 8000),
        ("encoding", "H265"),
        ("udp_buffer_size_bytes", 0),
        ("udp_buffer_size_bytes", 2**31),
        ("decoder", "vtdec_hw ! filesink location=/tmp/bad"),
        ("decoder", ""),
        ("decoder", None),
        ("expected_width_px", 0),
        ("expected_height_px", 9000),
        ("expected_fps", float("nan")),
        ("expected_fps", 0),
        ("timeout_s", float("inf")),
        ("timeout_s", 0),
        ("timeout_s", True),
        ("max_frames", 0),
        ("max_frames", True),
        ("max_frames", 1.2),
    ],
)
def test_invalid_config(key, value):
    with pytest.raises(ValueError, match=key):
        RtpConfig(**{key: value})


def test_pipeline_and_committed_config():
    config = load_rtp_config(Path("config/live_apfpv.yaml"))
    assert config == RtpConfig()
    pipeline = build_pipeline(
        RtpConfig(port=5678, decoder="avdec_h264", payload_type=97)
    )
    assert "port=5678" in pipeline
    assert "payload=97" in pipeline
    assert "! rtph264depay ! h264parse !" in pipeline
    assert "! avdec_h264 !" in pipeline
    assert pipeline.count("max-size-buffers=1") == 2
    assert pipeline.count("leaky=downstream") == 2
    assert "video/x-raw,format=BGR" in pipeline
    assert "max-buffers=1 drop=true sync=false" in pipeline
    assert "wait-on-eos=false" in pipeline
    assert "vtdec_hw" in build_pipeline(config)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        "camera: []",
        "camera: {source: images}",
        "camera: {source: rtp, typo: 1}",
        "camera: {source: rtp, rtp: {pot: 5600}}",
        "camera: {source: rtp, expected: {width: 1920}}",
        "camera: {source: rtp, rtp: {port: true}}",
        "camera: {source: rtp, source: rtp}",
        "camera: &c {source: rtp, rtp: *c}",
        "!!python/object/apply:os.system ['echo unsafe']",
        "[",
        "x" * 65537,
    ],
)
def test_bad_yaml(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_rtp_config(path)


def test_yaml_defaults_overrides_and_missing(tmp_path):
    path = tmp_path / "live.yaml"
    with pytest.raises(ValueError, match="cannot read"):
        load_rtp_config(path)
    path.write_text("camera: {source: rtp, rtp: {port: 1234}, expected: {fps: 25}}")
    assert load_rtp_config(path) == RtpConfig(port=1234, expected_fps=25)
