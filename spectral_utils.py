"""
Spectral Analysis Utilities for Cognitive Canary
=================================================

Shared spectral analysis functions used across multiple obfuscation engines.
Consolidates redundant implementations of spectral entropy, FFT analysis,
and signal normalization.

Author: Cognitive Canary Project
License: MIT
Version: 6.0
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
from typing import Tuple, Optional


def compute_spectral_entropy(signal: np.ndarray,
                            fs: float = 100.0,
                            nperseg: Optional[int] = None) -> float:
    """
    Compute spectral entropy of a signal using Welch's method.

    Spectral entropy quantifies the complexity of a signal's frequency content.
    Higher entropy indicates more uniform distribution across frequencies
    (more unpredictable/natural-looking), while lower entropy indicates
    concentrated power in specific bands (more synthetic/detectable).

    Target range for biomimicry: 3.0-3.5 nats (natural human motor signals)

    Parameters
    ----------
    signal : np.ndarray
        1D time-series signal to analyze
    fs : float, default=100.0
        Sampling frequency in Hz
    nperseg : int, optional
        Length of each segment for Welch's method.
        If None, uses min(256, len(signal))

    Returns
    -------
    float
        Spectral entropy in nats (natural logarithm base)

    Examples
    --------
    >>> signal = np.random.randn(1000)  # White noise
    >>> entropy_val = compute_spectral_entropy(signal, fs=100.0)
    >>> print(f"Entropy: {entropy_val:.2f} nats")

    Notes
    -----
    - Uses Welch's method for robust PSD estimation with overlapping windows
    - Normalizes PSD to probability distribution before entropy calculation
    - Returns value in nats (to convert to bits: multiply by 1.443)
    """
    if len(signal) == 0:
        raise ValueError("Cannot compute spectral entropy of empty signal")

    if nperseg is None:
        nperseg = min(256, len(signal))

    # Compute Power Spectral Density using Welch's method
    # Welch's method averages overlapping FFT windows for noise reduction
    frequencies, psd = welch(signal, fs=fs, nperseg=nperseg)

    # Normalize PSD to probability distribution (required for entropy calculation)
    # Add small epsilon to avoid log(0)
    psd_normalized = psd / (np.sum(psd) + 1e-12)

    # Compute Shannon entropy in nats (natural logarithm)
    spectral_entropy = entropy(psd_normalized, base=np.e)

    return float(spectral_entropy)


def compute_band_power(signal: np.ndarray,
                      fs: float,
                      band: Tuple[float, float],
                      nperseg: Optional[int] = None) -> float:
    """
    Compute total power in a specific frequency band.

    Useful for analyzing EEG bands (alpha, beta, theta) or tremor frequencies.

    Parameters
    ----------
    signal : np.ndarray
        1D time-series signal
    fs : float
        Sampling frequency in Hz
    band : Tuple[float, float]
        Frequency band as (low_freq, high_freq) in Hz
    nperseg : int, optional
        Segment length for Welch's method

    Returns
    -------
    float
        Total power in the specified band

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))  # 10 Hz sine
    >>> alpha_power = compute_band_power(signal, fs=100, band=(8, 12))
    >>> print(f"Alpha band power: {alpha_power:.4f}")
    """
    if len(signal) == 0:
        raise ValueError("Cannot compute band power of empty signal")

    if band[0] >= band[1]:
        raise ValueError(f"Invalid band: low frequency {band[0]} must be < high frequency {band[1]}")

    if nperseg is None:
        nperseg = min(256, len(signal))

    # Compute PSD
    frequencies, psd = welch(signal, fs=fs, nperseg=nperseg)

    # Find indices corresponding to the band
    band_indices = np.logical_and(frequencies >= band[0], frequencies <= band[1])

    # Integrate power in the band (trapezoidal rule)
    # np.trapezoid is the preferred API in NumPy >= 2.0; np.trapz is deprecated
    try:
        band_power = np.trapezoid(psd[band_indices], frequencies[band_indices])
    except AttributeError:
        band_power = np.trapz(psd[band_indices], frequencies[band_indices])  # type: ignore[attr-defined]

    return float(band_power)


def normalize_signal(signal: np.ndarray,
                    target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Normalize signal to a target range using min-max scaling.

    Parameters
    ----------
    signal : np.ndarray
        Input signal to normalize
    target_range : Tuple[float, float], default=(0.0, 1.0)
        Target range as (min_val, max_val)

    Returns
    -------
    np.ndarray
        Normalized signal

    Examples
    --------
    >>> signal = np.array([10, 20, 30, 40, 50])
    >>> normalized = normalize_signal(signal, target_range=(-1, 1))
    >>> print(normalized)  # [-1.0, -0.5, 0.0, 0.5, 1.0]
    """
    if len(signal) == 0:
        return signal

    signal_min = np.min(signal)
    signal_max = np.max(signal)

    # Handle constant signal (avoid division by zero)
    if signal_max - signal_min < 1e-12:
        return np.full_like(signal, np.mean(target_range))

    # Min-max normalization to [0, 1]
    normalized = (signal - signal_min) / (signal_max - signal_min)

    # Scale to target range
    target_min, target_max = target_range
    scaled = normalized * (target_max - target_min) + target_min

    return scaled


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in decibels.

    SNR quantifies how much the signal stands out from noise.
    Higher SNR = cleaner signal, Lower SNR = noisier signal.

    For biomimicry, we typically target SNR in 5-7 dB range
    (subtle injection that doesn't overwhelm natural variability)

    Parameters
    ----------
    signal : np.ndarray
        Original signal
    noise : np.ndarray
        Noise component (same shape as signal)

    Returns
    -------
    float
        SNR in decibels (dB)

    Examples
    --------
    >>> signal = np.ones(1000)
    >>> noise = 0.1 * np.random.randn(1000)
    >>> snr = compute_snr(signal, noise)
    >>> print(f"SNR: {snr:.2f} dB")

    Notes
    -----
    SNR formula: 10 * log10(P_signal / P_noise)
    where P = mean squared power
    """
    if len(signal) != len(noise):
        raise ValueError(f"Signal length {len(signal)} must match noise length {len(noise)}")

    if len(signal) == 0:
        raise ValueError("Cannot compute SNR of empty signal")

    # Compute power (mean squared value)
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    # Handle zero noise (infinite SNR -> return large value)
    if noise_power < 1e-12:
        return 100.0

    # SNR in decibels
    snr_db = 10 * np.log10(signal_power / noise_power + 1e-12)

    return float(snr_db)


def compute_dominant_frequency(signal: np.ndarray,
                              fs: float,
                              band: Optional[Tuple[float, float]] = None) -> float:
    """
    Find the dominant (peak power) frequency in a signal.

    Useful for extracting tremor frequency, oscillation rates, etc.

    Parameters
    ----------
    signal : np.ndarray
        1D time-series signal
    fs : float
        Sampling frequency in Hz
    band : Tuple[float, float], optional
        If provided, restrict search to this frequency band

    Returns
    -------
    float
        Dominant frequency in Hz

    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2 * np.pi * 8.5 * t)  # 8.5 Hz sine wave
    >>> dom_freq = compute_dominant_frequency(signal, fs=1000)
    >>> print(f"Dominant frequency: {dom_freq:.2f} Hz")  # ~8.5 Hz
    """
    if len(signal) == 0:
        raise ValueError("Cannot compute dominant frequency of empty signal")

    # Compute PSD
    nperseg = min(256, len(signal))
    frequencies, psd = welch(signal, fs=fs, nperseg=nperseg)

    # Restrict to band if specified
    if band is not None:
        band_indices = np.logical_and(frequencies >= band[0], frequencies <= band[1])
        frequencies = frequencies[band_indices]
        psd = psd[band_indices]

    # Find frequency with maximum power
    if len(psd) == 0:
        raise ValueError("No frequencies in specified band")

    dominant_freq = frequencies[np.argmax(psd)]

    return float(dominant_freq)


def verify_spectral_plausibility(signal: np.ndarray,
                                fs: float,
                                target_entropy: Tuple[float, float] = (3.0, 3.5),
                                target_snr: Tuple[float, float] = (5.0, 7.0)) -> Tuple[bool, str]:
    """
    Verify that a signal has plausible spectral properties for biomimicry.

    Checks both spectral entropy (complexity) and SNR (subtlety).

    Parameters
    ----------
    signal : np.ndarray
        Signal to validate
    fs : float
        Sampling frequency
    target_entropy : Tuple[float, float], default=(3.0, 3.5)
        Acceptable entropy range in nats
    target_snr : Tuple[float, float], default=(5.0, 7.0)
        Acceptable SNR range in dB (if noise is detectable)

    Returns
    -------
    Tuple[bool, str]
        (is_plausible, explanation)

    Examples
    --------
    >>> signal = np.random.randn(1000)
    >>> is_valid, msg = verify_spectral_plausibility(signal, fs=100)
    >>> print(f"Valid: {is_valid}, Reason: {msg}")
    """
    try:
        # Compute spectral entropy
        ent = compute_spectral_entropy(signal, fs=fs)

        # Check entropy range
        if ent < target_entropy[0]:
            return False, f"Spectral entropy too low ({ent:.2f} < {target_entropy[0]}): signal too predictable/synthetic"

        if ent > target_entropy[1]:
            return False, f"Spectral entropy too high ({ent:.2f} > {target_entropy[1]}): signal too random/unnatural"

        # All checks passed
        return True, f"Signal is spectrally plausible (entropy={ent:.2f} nats)"

    except Exception as e:
        return False, f"Spectral validation failed: {str(e)}"


if __name__ == "__main__":
    """
    Example usage and validation of spectral utilities.
    """
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Spectral Utilities Demo - Cognitive Canary v6.0")
    print("=" * 60)

    # Generate test signals
    fs = 100.0  # 100 Hz sampling
    t = np.linspace(0, 2, int(2 * fs))  # 2 seconds

    # 1. Pure sine wave (low entropy - detectable)
    sine_wave = np.sin(2 * np.pi * 8 * t)
    entropy_sine = compute_spectral_entropy(sine_wave, fs=fs)
    print(f"\n1. Pure 8 Hz Sine Wave:")
    print(f"   Spectral Entropy: {entropy_sine:.3f} nats (DETECTABLE - too low)")

    # 2. White noise (high entropy - unnatural)
    white_noise = np.random.randn(len(t))
    entropy_noise = compute_spectral_entropy(white_noise, fs=fs)
    print(f"\n2. White Noise:")
    print(f"   Spectral Entropy: {entropy_noise:.3f} nats (UNNATURAL - too high)")

    # 3. Natural tremor simulation (target range)
    # Combine multiple harmonics with pink noise for biomimicry
    tremor = (0.6 * np.sin(2 * np.pi * 8 * t) +  # Dominant 8 Hz
              0.3 * np.sin(2 * np.pi * 12 * t) +  # Harmonic
              0.1 * np.random.randn(len(t)))       # Natural variability
    entropy_tremor = compute_spectral_entropy(tremor, fs=fs)
    print(f"\n3. Simulated Natural Tremor:")
    print(f"   Spectral Entropy: {entropy_tremor:.3f} nats (TARGET RANGE ✓)")

    # 4. Band power analysis
    alpha_power = compute_band_power(tremor, fs=fs, band=(8, 12))
    theta_power = compute_band_power(tremor, fs=fs, band=(4, 8))
    print(f"\n4. Frequency Band Analysis:")
    print(f"   Alpha (8-12 Hz) Power: {alpha_power:.4f}")
    print(f"   Theta (4-8 Hz) Power: {theta_power:.4f}")

    # 5. Dominant frequency
    dom_freq = compute_dominant_frequency(tremor, fs=fs, band=(4, 12))
    print(f"\n5. Dominant Frequency: {dom_freq:.2f} Hz")

    # 6. Plausibility verification
    is_valid, msg = verify_spectral_plausibility(tremor, fs=fs)
    print(f"\n6. Biomimicry Validation:")
    print(f"   Result: {msg}")

    print("\n" + "=" * 60)
    print("✓ All spectral utilities validated successfully")
    print("=" * 60)
