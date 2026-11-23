"""Frame arithmetic against the original analysis code. Skipped without the original code."""

from __future__ import annotations

from models import tiny_model
from ssl_silence import encoders as E


def test_frame_arithmetic(original, parity_record):
    O = original("features")
    kernels, strides = E.conv_layers(tiny_model("wavlm", n_layers=1))
    lengths = list(range(0, 1_300)) + list(range(63_600, 64_400)) + [8_000, 20_001, 130_000]
    worst = 0.0 if all(E.conv_output_length(n, kernels, strides) == O.conv_output_length(n)
                       for n in lengths) else float("inf")
    parity_record("encoders", "conv_output_length", len(lengths), worst)
    assert worst == 0.0
    cases = [(frames, n) for frames in (1, 2, 50, 199, 200) for n in lengths]
    worst = 0.0 if all(E.real_frame_count(f, n, kernels, strides) == O.real_frame_count(f, n)
                       for f, n in cases) else float("inf")
    parity_record("encoders", "real_frame_count", len(cases), worst)
    assert worst == 0.0
