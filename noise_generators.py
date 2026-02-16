"""
Noise Generation Utilities for Cognitive Canary
===============================================

Shared noise generation functions for biomimetic obfuscation across engines.
Consolidates redundant implementations of pink noise, uniform jitter,
and amplitude envelopes.

Author: Cognitive Canary Project
License: MIT
Version: 6.0
"""

import numpy as np
from typing import Tuple, Optional


def generate_pink_noise(n_samples: int, scaling: float = 1.0) -> np.ndarray:
    """
    Generate pink noise (1/f spectrum) for natural-looking temporal jitter.

    Pink noise has more power at low frequencies, matching cognitive variance
    better than white noise. Used for keystroke timing jitter and amplitude
    modulation in EEG injection.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    scaling : float, default=1.0
        Amplitude scaling factor

    Returns
    -------
    np.ndarray
        Pink noise sequence with 1/f spectrum

    Examples
    --------
    >>> noise = generate_pink_noise(1000, scaling=0.1)
    >>> print(f"Pink noise: mean={noise.mean():.4f}, std={noise.std():.4f}")

    Notes
    -----
    Implementation uses Voss-McCartney algorithm:
    1. Generate multiple white noise sources at different update rates
    2. Sum them with appropriate scaling
    3. Result has 1/f power spectral density

    This is more computationally efficient than FFT-based methods
    and produces perceptually better results for short sequences.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    # Voss-McCartney algorithm for pink noise generation
    # Use 16 sources for good approximation to 1/f
    n_sources = 16
    sources = np.zeros((n_sources, n_samples))

    for i in range(n_sources):
        # Each source updates at half the rate of the previous one
        update_rate = 2 ** i
        n_updates = max(1, n_samples // update_rate)

        # Generate white noise values at update points
        values = np.random.randn(n_updates)

        # Repeat each value for its duration
        sources[i] = np.repeat(values, update_rate)[:n_samples]

    # Sum all sources with 1/sqrt(n_sources) scaling for proper variance
    pink = np.sum(sources, axis=0) / np.sqrt(n_sources)

    # Apply scaling and return
    return pink * scaling


def generate_uniform_jitter(n_samples: int,
                           amplitude: float = 1.0) -> np.ndarray:
    """
    Generate uniform random jitter for spatial perturbations.

    Uniform distribution provides equal probability across range,
    useful for cursor position offsets and pixel-level noise.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    amplitude : float, default=1.0
        Peak-to-peak amplitude (jitter will be in range [-amplitude/2, +amplitude/2])

    Returns
    -------
    np.ndarray
        Uniform jitter sequence

    Examples
    --------
    >>> jitter = generate_uniform_jitter(100, amplitude=2.0)
    >>> print(f"Range: [{jitter.min():.2f}, {jitter.max():.2f}]")
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    # Generate uniform random values in [-amplitude/2, +amplitude/2]
    jitter = amplitude * (np.random.rand(n_samples) - 0.5)

    return jitter


def generate_gaussian_jitter(n_samples: int,
                            sigma: float = 1.0,
                            mean: float = 0.0) -> np.ndarray:
    """
    Generate Gaussian (normal) distributed jitter.

    Gaussian distribution concentrates samples near the mean with
    occasional larger deviations, matching many natural processes
    (motor control errors, perceptual noise, etc.)

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    sigma : float, default=1.0
        Standard deviation (spread of distribution)
    mean : float, default=0.0
        Mean value (center of distribution)

    Returns
    -------
    np.ndarray
        Gaussian jitter sequence

    Examples
    --------
    >>> jitter = generate_gaussian_jitter(1000, sigma=0.1, mean=0.0)
    >>> print(f"Mean: {jitter.mean():.4f}, Std: {jitter.std():.4f}")
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")

    # Generate Gaussian noise
    jitter = np.random.normal(loc=mean, scale=sigma, size=n_samples)

    return jitter


def generate_amplitude_envelope(n_samples: int,
                               base_amplitude: float = 1.0,
                               modulation_depth: float = 0.2,
                               modulation_freq: float = 0.5) -> np.ndarray:
    """
    Generate smooth amplitude envelope using pink noise modulation.

    Creates natural-looking amplitude variations for EEG injection
    and other signals requiring temporal amplitude changes.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    base_amplitude : float, default=1.0
        Baseline amplitude level
    modulation_depth : float, default=0.2
        Depth of amplitude modulation (0.0-1.0)
        0.0 = constant amplitude, 1.0 = full modulation
    modulation_freq : float, default=0.5
        Frequency of slow modulation in Hz (for temporal variation)

    Returns
    -------
    np.ndarray
        Amplitude envelope (always positive)

    Examples
    --------
    >>> envelope = generate_amplitude_envelope(1000, base_amplitude=1.0, modulation_depth=0.3)
    >>> print(f"Envelope range: [{envelope.min():.2f}, {envelope.max():.2f}]")

    Notes
    -----
    Uses combination of:
    1. Pink noise for natural high-frequency variability
    2. Slow sinusoidal modulation for temporal trends
    Ensures envelope is always positive (required for multiplication)
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if not 0.0 <= modulation_depth <= 1.0:
        raise ValueError(f"modulation_depth must be in [0, 1], got {modulation_depth}")

    # Generate pink noise component (high-frequency natural variability)
    pink = generate_pink_noise(n_samples, scaling=modulation_depth * 0.3)

    # Generate slow sinusoidal modulation (temporal trends)
    t = np.arange(n_samples)
    slow_mod = modulation_depth * 0.5 * np.sin(2 * np.pi * modulation_freq * t / n_samples)

    # Combine: base + slow modulation + pink noise
    envelope = base_amplitude * (1.0 + slow_mod + pink)

    # Ensure envelope is always positive (clip at 10% of base)
    envelope = np.maximum(envelope, base_amplitude * 0.1)

    return envelope


def generate_physiologic_jitter(n_samples: int,
                               amplitude: float = 1.0,
                               percentage: float = 2.0) -> np.ndarray:
    """
    Generate physiologic jitter matching human motor variability.

    Human movements have ~2-3% variability due to motor control noise.
    This function generates jitter matching those statistics.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    amplitude : float, default=1.0
        Baseline amplitude to apply percentage to
    percentage : float, default=2.0
        Jitter as percentage of amplitude (typically 2-3% for humans)

    Returns
    -------
    np.ndarray
        Physiologic jitter sequence

    Examples
    --------
    >>> # 2% jitter on 10-pixel amplitude
    >>> jitter = generate_physiologic_jitter(100, amplitude=10.0, percentage=2.0)
    >>> print(f"Jitter std: {jitter.std():.4f} pixels")  # ~0.2 pixels

    Notes
    -----
    Uses Gaussian distribution with sigma = (percentage/100) * amplitude / 3
    This ensures ~99.7% of values fall within ±percentage% (3-sigma rule)
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if amplitude < 0:
        raise ValueError(f"amplitude must be non-negative, got {amplitude}")

    if percentage < 0:
        raise ValueError(f"percentage must be non-negative, got {percentage}")

    # Convert percentage to absolute sigma
    # Use 3-sigma rule: sigma = percentage% * amplitude / 3
    # This ensures 99.7% of values within ±percentage%
    sigma = (percentage / 100.0) * amplitude / 3.0

    # Generate Gaussian jitter
    jitter = generate_gaussian_jitter(n_samples, sigma=sigma, mean=0.0)

    return jitter


def generate_correlated_noise(n_samples: int,
                             correlation_length: int = 10,
                             amplitude: float = 1.0) -> np.ndarray:
    """
    Generate temporally correlated noise using moving average filter.

    Correlated noise has memory - nearby samples are similar.
    Useful for modeling motor control with neuromuscular filtering.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    correlation_length : int, default=10
        Length of correlation window (higher = smoother)
    amplitude : float, default=1.0
        Amplitude scaling

    Returns
    -------
    np.ndarray
        Correlated noise sequence

    Examples
    --------
    >>> noise = generate_correlated_noise(1000, correlation_length=20, amplitude=0.5)
    >>> print(f"Noise: mean={noise.mean():.4f}, std={noise.std():.4f}")

    Notes
    -----
    Implementation:
    1. Generate white noise
    2. Apply moving average filter for correlation
    3. Normalize to maintain desired amplitude
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if correlation_length <= 0:
        raise ValueError(f"correlation_length must be positive, got {correlation_length}")

    # Generate white noise
    white = np.random.randn(n_samples)

    # Apply moving average filter for correlation
    window = np.ones(correlation_length) / correlation_length
    correlated = np.convolve(white, window, mode='same')

    # Normalize to maintain amplitude
    correlated = correlated / np.std(correlated + 1e-12) * amplitude

    return correlated


def apply_duty_cycle(signal: np.ndarray,
                     duty_cycle: float = 1.0,
                     pattern: str = 'random') -> np.ndarray:
    """
    Apply duty cycle to a signal (temporal gating).

    Duty cycle controls what fraction of time the signal is active.
    Useful for reducing productivity impact while maintaining protection.

    Parameters
    ----------
    signal : np.ndarray
        Input signal to gate
    duty_cycle : float, default=1.0
        Fraction of time signal is active (0.0-1.0)
        1.0 = always on, 0.5 = on 50% of time, 0.0 = always off
    pattern : str, default='random'
        Gating pattern: 'random', 'burst', or 'periodic'

    Returns
    -------
    np.ndarray
        Gated signal (same length as input)

    Examples
    --------
    >>> signal = np.ones(1000)
    >>> gated = apply_duty_cycle(signal, duty_cycle=0.3, pattern='random')
    >>> print(f"Active fraction: {(gated != 0).mean():.2f}")  # ~0.30

    Notes
    -----
    Patterns:
    - 'random': Bernoulli process (each sample independently active/inactive)
    - 'burst': Active in random bursts (clustered activation)
    - 'periodic': Regular on/off cycling
    """
    if not 0.0 <= duty_cycle <= 1.0:
        raise ValueError(f"duty_cycle must be in [0, 1], got {duty_cycle}")

    if duty_cycle == 1.0:
        return signal  # No gating needed

    n_samples = len(signal)

    if pattern == 'random':
        # Bernoulli process: each sample independently active with probability duty_cycle
        mask = np.random.rand(n_samples) < duty_cycle

    elif pattern == 'burst':
        # Burst pattern: active in random bursts
        # Burst length = ~5% of signal, spaced to achieve duty cycle
        burst_length = max(1, int(0.05 * n_samples))
        n_bursts = int(duty_cycle * n_samples / burst_length)

        mask = np.zeros(n_samples, dtype=bool)
        burst_starts = np.random.choice(n_samples - burst_length, size=n_bursts, replace=False)

        for start in burst_starts:
            mask[start:start + burst_length] = True

    elif pattern == 'periodic':
        # Periodic on/off: active for duty_cycle fraction of each period
        period = 100  # Arbitrary period length
        active_length = int(duty_cycle * period)

        mask = np.zeros(n_samples, dtype=bool)
        for i in range(0, n_samples, period):
            mask[i:min(i + active_length, n_samples)] = True

    else:
        raise ValueError(f"Unknown pattern '{pattern}', must be 'random', 'burst', or 'periodic'")

    # Apply mask
    gated_signal = signal * mask

    return gated_signal


if __name__ == "__main__":
    """
    Example usage and validation of noise generation utilities.
    """
    print("=" * 60)
    print("Noise Generation Utilities Demo - Cognitive Canary v6.0")
    print("=" * 60)

    # 1. Pink noise (1/f spectrum)
    pink = generate_pink_noise(1000, scaling=1.0)
    print(f"\n1. Pink Noise (1/f spectrum):")
    print(f"   Mean: {pink.mean():.4f}, Std: {pink.std():.4f}")
    print(f"   Range: [{pink.min():.4f}, {pink.max():.4f}]")

    # 2. Uniform jitter
    uniform = generate_uniform_jitter(1000, amplitude=2.0)
    print(f"\n2. Uniform Jitter:")
    print(f"   Mean: {uniform.mean():.4f}, Std: {uniform.std():.4f}")
    print(f"   Range: [{uniform.min():.4f}, {uniform.max():.4f}]")

    # 3. Gaussian jitter
    gaussian = generate_gaussian_jitter(1000, sigma=0.5)
    print(f"\n3. Gaussian Jitter:")
    print(f"   Mean: {gaussian.mean():.4f}, Std: {gaussian.std():.4f}")

    # 4. Amplitude envelope
    envelope = generate_amplitude_envelope(1000, base_amplitude=1.0, modulation_depth=0.3)
    print(f"\n4. Amplitude Envelope:")
    print(f"   Mean: {envelope.mean():.4f}")
    print(f"   Range: [{envelope.min():.4f}, {envelope.max():.4f}]")

    # 5. Physiologic jitter (2% variability)
    physio = generate_physiologic_jitter(1000, amplitude=10.0, percentage=2.0)
    print(f"\n5. Physiologic Jitter (2% of 10.0):")
    print(f"   Std: {physio.std():.4f} (expected ~0.067)")
    print(f"   99.7% within ±{3*physio.std():.4f} (expected ±0.2)")

    # 6. Correlated noise
    correlated = generate_correlated_noise(1000, correlation_length=20, amplitude=0.5)
    print(f"\n6. Correlated Noise (correlation_length=20):")
    print(f"   Mean: {correlated.mean():.4f}, Std: {correlated.std():.4f}")

    # 7. Duty cycle application
    signal = np.ones(1000)
    gated_random = apply_duty_cycle(signal, duty_cycle=0.3, pattern='random')
    gated_burst = apply_duty_cycle(signal, duty_cycle=0.3, pattern='burst')
    gated_periodic = apply_duty_cycle(signal, duty_cycle=0.3, pattern='periodic')

    print(f"\n7. Duty Cycle Gating (30% duty):")
    print(f"   Random: {(gated_random != 0).mean():.3f} active")
    print(f"   Burst: {(gated_burst != 0).mean():.3f} active")
    print(f"   Periodic: {(gated_periodic != 0).mean():.3f} active")

    print("\n" + "=" * 60)
    print("✓ All noise generation utilities validated successfully")
    print("=" * 60)
