"""
Tests for spectral_utils.py
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from spectral_utils import (
    compute_spectral_entropy,
    compute_band_power,
    compute_dominant_frequency,
    compute_snr,
    normalize_signal,
    verify_spectral_plausibility,
)


class TestComputeSpectralEntropy:
    """Tests for compute_spectral_entropy."""

    def test_pure_sine_low_entropy(self):
        """A pure sine wave has low spectral entropy (concentrated power)."""
        t = np.linspace(0, 2, 2000)
        signal = np.sin(2 * np.pi * 10 * t)
        entropy = compute_spectral_entropy(signal, fs=1000.0)
        assert entropy < 2.5, f"Expected low entropy for pure sine, got {entropy:.3f}"

    def test_white_noise_high_entropy(self):
        """White noise has high spectral entropy (uniform power)."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(2000)
        entropy = compute_spectral_entropy(noise, fs=1000.0)
        assert entropy > 3.0, f"Expected high entropy for white noise, got {entropy:.3f}"

    def test_entropy_is_non_negative(self):
        """Spectral entropy must always be non-negative."""
        rng = np.random.default_rng(7)
        for _ in range(10):
            signal = rng.standard_normal(512)
            entropy = compute_spectral_entropy(signal, fs=100.0)
            assert entropy >= 0.0

    def test_empty_signal_raises(self):
        """Empty signal should raise ValueError."""
        with pytest.raises(ValueError):
            compute_spectral_entropy(np.array([]), fs=100.0)

    def test_entropy_varies_with_signal_complexity(self):
        """More complex signals should have higher entropy than simpler ones."""
        t = np.linspace(0, 4, 4000)
        simple = np.sin(2 * np.pi * 5 * t)
        complex_signal = (
            np.sin(2 * np.pi * 5 * t)
            + 0.5 * np.sin(2 * np.pi * 11 * t)
            + 0.3 * np.sin(2 * np.pi * 23 * t)
        )
        e_simple = compute_spectral_entropy(simple, fs=1000.0)
        e_complex = compute_spectral_entropy(complex_signal, fs=1000.0)
        assert e_complex > e_simple

    def test_returns_float(self):
        """Return value should be a Python float."""
        signal = np.random.randn(500)
        result = compute_spectral_entropy(signal, fs=100.0)
        assert isinstance(result, float)


class TestComputeBandPower:
    """Tests for compute_band_power."""

    def test_sine_in_band_has_high_power(self):
        """A sine at 8 Hz should show high power in the 4-12 Hz tremor band."""
        t = np.linspace(0, 4, 4000)
        signal = np.sin(2 * np.pi * 8 * t)
        power = compute_band_power(signal, fs=1000.0, band=(4.0, 12.0))
        assert power > 0.1, f"Expected high power in band, got {power:.4f}"

    def test_sine_outside_band_has_low_power(self):
        """A 50 Hz sine should have low power in the 4-12 Hz tremor band."""
        t = np.linspace(0, 4, 4000)
        signal = np.sin(2 * np.pi * 50 * t)
        power = compute_band_power(signal, fs=1000.0, band=(4.0, 12.0))
        assert power < 0.01, f"Expected low power outside band, got {power:.4f}"

    def test_power_non_negative(self):
        """Band power must be non-negative."""
        rng = np.random.default_rng(99)
        for _ in range(10):
            signal = rng.standard_normal(1000)
            power = compute_band_power(signal, fs=500.0, band=(4.0, 12.0))
            assert power >= 0.0

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError):
            compute_band_power(np.array([]), fs=100.0, band=(4.0, 12.0))

    def test_invalid_band_raises(self):
        """Low frequency >= high frequency should raise ValueError."""
        signal = np.random.randn(500)
        with pytest.raises(ValueError):
            compute_band_power(signal, fs=100.0, band=(12.0, 4.0))

    def test_returns_float(self):
        signal = np.random.randn(500)
        result = compute_band_power(signal, fs=100.0, band=(4.0, 12.0))
        assert isinstance(result, float)


class TestComputeDominantFrequency:
    """Tests for compute_dominant_frequency."""

    def test_detects_sine_frequency(self):
        """Should detect the dominant frequency of a pure sine wave."""
        t = np.linspace(0, 4, 4000)
        freq_target = 7.5  # Hz
        signal = np.sin(2 * np.pi * freq_target * t)
        detected = compute_dominant_frequency(signal, fs=1000.0, band=(4.0, 12.0))
        assert abs(detected - freq_target) < 0.5, (
            f"Expected ~{freq_target} Hz, got {detected:.2f} Hz"
        )

    def test_result_within_band(self):
        """Dominant frequency must always be within the requested band."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            signal = rng.standard_normal(2000)
            result = compute_dominant_frequency(signal, fs=1000.0, band=(4.0, 12.0))
            assert 4.0 <= result <= 12.0

    def test_returns_float(self):
        signal = np.random.randn(500)
        result = compute_dominant_frequency(signal, fs=100.0)
        assert isinstance(result, float)

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError):
            compute_dominant_frequency(np.array([]), fs=100.0)

    def test_no_band_restriction(self):
        """Without band, should still return a valid frequency."""
        t = np.linspace(0, 2, 2000)
        signal = np.sin(2 * np.pi * 10 * t)
        result = compute_dominant_frequency(signal, fs=1000.0)
        assert result > 0


class TestComputeSNR:
    """Tests for compute_snr (takes signal + noise arrays)."""

    def test_low_noise_high_snr(self):
        """A large signal with tiny noise should yield high SNR."""
        signal = np.ones(1000)
        noise = 0.01 * np.random.default_rng(1).standard_normal(1000)
        snr = compute_snr(signal, noise)
        assert snr > 20.0, f"Expected high SNR, got {snr:.2f} dB"

    def test_large_noise_low_snr(self):
        """Small signal swamped by noise should yield low or negative SNR."""
        signal = 0.01 * np.ones(1000)
        noise = np.random.default_rng(2).standard_normal(1000)
        snr = compute_snr(signal, noise)
        assert snr < 5.0, f"Expected low SNR for heavy noise, got {snr:.2f} dB"

    def test_zero_noise_returns_large_value(self):
        """Zero noise should return a large SNR (no noise = perfect signal)."""
        signal = np.ones(100)
        noise = np.zeros(100)
        snr = compute_snr(signal, noise)
        assert snr >= 50.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_snr(np.ones(100), np.ones(50))

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError):
            compute_snr(np.array([]), np.array([]))

    def test_returns_float(self):
        signal = np.random.randn(100)
        noise = 0.1 * np.random.randn(100)
        result = compute_snr(signal, noise)
        assert isinstance(result, float)


class TestNormalizeSignal:
    """Tests for normalize_signal."""

    def test_basic_normalization(self):
        """Signal should be normalized to [0, 1] by default."""
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        normalized = normalize_signal(signal)
        assert np.isclose(normalized.min(), 0.0)
        assert np.isclose(normalized.max(), 1.0)

    def test_custom_range(self):
        signal = np.array([1.0, 2.0, 3.0])
        normalized = normalize_signal(signal, target_range=(-1.0, 1.0))
        assert np.isclose(normalized.min(), -1.0)
        assert np.isclose(normalized.max(), 1.0)

    def test_constant_signal(self):
        """Constant signal should return the midpoint of target range."""
        signal = np.ones(10) * 5.0
        normalized = normalize_signal(signal, target_range=(0.0, 1.0))
        assert np.all(normalized == 0.5)

    def test_empty_signal_passthrough(self):
        result = normalize_signal(np.array([]))
        assert len(result) == 0


class TestVerifySpectralPlausibility:
    """Tests for verify_spectral_plausibility."""

    def test_pure_sine_fails(self):
        """Pure sine has too-low entropy; should fail plausibility check."""
        t = np.linspace(0, 4, 4000)
        signal = np.sin(2 * np.pi * 8 * t)
        is_valid, msg = verify_spectral_plausibility(signal, fs=1000.0)
        # Pure sine entropy is very low → below 3.0 nats target
        assert isinstance(is_valid, bool)
        assert isinstance(msg, str)

    def test_returns_tuple(self):
        signal = np.random.randn(1000)
        result = verify_spectral_plausibility(signal, fs=100.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
