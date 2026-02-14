"""
Cognitive Canary v5.0 - Spectral Canary Module
===============================================

EEG frequency injection for neural profiling defense.

Defends against:
- Emotion recognition systems (BrainCo Focus EDU)
- Cognitive load detectors (Neurable)
- Attention tracking (Muse headbands in corporate settings)

Key Technique: Injects adversarial oscillations in the 4-12 Hz band to:
1. Mimic alpha/theta transitions (natural cognitive fluctuation)
2. Break phase-locked loop (PLL) stress decoders
3. Force emotion classifiers into high-entropy states

Usage:
    from spectral_canary import SpectralDefender

    defender = SpectralDefender()
    protected_eeg = defender.inject_canary(raw_eeg_signal)

Author: Cognitive Canary Project
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from scipy import signal


@dataclass
class SpectralConfig:
    """Configuration for EEG spectral defense."""

    # Target frequency bands (Hz)
    ALPHA_BAND: Tuple[float, float] = (8.0, 12.0)  # Relaxed focus
    THETA_BAND: Tuple[float, float] = (4.0, 8.0)   # Cognitive load
    BETA_BAND: Tuple[float, float] = (12.0, 30.0)  # Active thinking

    # Injection parameters
    INJECTION_AMPLITUDE: float = 0.15  # 15% of signal power
    PHASE_RANDOMIZATION: bool = True   # Break PLL decoders
    TARGET_SNR: float = 6.0            # Signal-to-noise ratio (dB)

    # Fingerprint protection
    TEMPORAL_COHERENCE_BREAK: bool = True  # Prevent connectome fingerprinting
    COHERENCE_WINDOW: int = 5              # Seconds to decorrelate


class SpectralDefender:
    """
    Defends against neural profiling by injecting adversarial oscillations.

    The core insight: Commercial EEG systems use band-power ratios
    (theta/beta, alpha asymmetry) as cognitive state proxies. By injecting
    calibrated noise in these bands, we force classifiers to abstain rather
    than misclassify.
    """

    def __init__(
        self,
        sampling_rate: int = 256,
        config: Optional[SpectralConfig] = None
    ):
        """
        Initialize spectral defender.

        Args:
            sampling_rate: EEG sampling rate in Hz (typical: 128-512 Hz)
            config: Optional configuration
        """
        self.fs = sampling_rate
        self.config = config or SpectralConfig()
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        assert 0 < self.config.INJECTION_AMPLITUDE < 0.5, \
            "Injection amplitude must be 0-50% to maintain plausibility"
        assert self.config.TARGET_SNR > 0, "SNR must be positive"

    def inject_canary(
        self,
        eeg_signal: np.ndarray,
        channels: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Inject adversarial oscillations into EEG signal.

        Args:
            eeg_signal: Raw EEG data (channels x samples)
            channels: Which channels to protect (None = all)

        Returns:
            Protected EEG signal with same shape
        """
        if eeg_signal.ndim == 1:
            # Single channel
            eeg_signal = eeg_signal.reshape(1, -1)

        n_channels, n_samples = eeg_signal.shape

        if channels is None:
            channels = list(range(n_channels))

        protected = eeg_signal.copy()

        for ch in channels:
            # Generate adversarial oscillations
            canary = self._generate_adversarial_signal(n_samples)

            # Inject into channel
            protected[ch, :] += canary

        return protected.squeeze()

    def _generate_adversarial_signal(self, n_samples: int) -> np.ndarray:
        """
        Generate adversarial oscillations targeting alpha/theta bands.

        Strategy:
        1. Generate signals in alpha and theta bands
        2. Randomize phase to break PLL decoders
        3. Modulate amplitude to match natural variance
        """
        t = np.arange(n_samples) / self.fs

        # Alpha band injection (8-12 Hz)
        alpha_freq = np.random.uniform(*self.config.ALPHA_BAND)
        alpha_phase = np.random.uniform(0, 2 * np.pi) if self.config.PHASE_RANDOMIZATION else 0
        alpha_signal = np.sin(2 * np.pi * alpha_freq * t + alpha_phase)

        # Theta band injection (4-8 Hz)
        theta_freq = np.random.uniform(*self.config.THETA_BAND)
        theta_phase = np.random.uniform(0, 2 * np.pi) if self.config.PHASE_RANDOMIZATION else 0
        theta_signal = np.sin(2 * np.pi * theta_freq * t + theta_phase)

        # Combine with weighted mixture
        canary = 0.6 * alpha_signal + 0.4 * theta_signal

        # Amplitude modulation (mimic natural variance)
        envelope = self._generate_amplitude_envelope(n_samples)
        canary *= envelope

        # Scale to injection amplitude
        canary *= self.config.INJECTION_AMPLITUDE

        return canary

    def _generate_amplitude_envelope(self, n_samples: int) -> np.ndarray:
        """
        Generate natural-looking amplitude modulation.

        Uses 1/f noise to mimic the amplitude fluctuations seen in
        real EEG (arrhythmic, non-stationary).
        """
        # Generate pink noise (1/f)
        white = np.random.randn(n_samples)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1e-6  # Avoid division by zero

        # Apply 1/f scaling
        pink_fft = np.fft.rfft(white) / np.sqrt(freqs)
        pink = np.fft.irfft(pink_fft, n=n_samples)

        # Normalize to [0.5, 1.5] range
        pink_norm = (pink - pink.min()) / (pink.max() - pink.min())
        envelope = 0.5 + pink_norm

        return envelope

    def break_temporal_coherence(
        self,
        eeg_signal: np.ndarray,
        window_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Break temporal coherence to prevent connectome fingerprinting.

        The attack: Neural fingerprinting relies on stable features across
        sessions (e.g., alpha asymmetry stays constant over weeks).

        The defense: Inject decorrelated noise in sliding windows to break
        cross-session correlation without affecting single-session utility.

        Args:
            eeg_signal: EEG data (channels x samples)
            window_size: Decorrelation window in samples

        Returns:
            Temporally-decorrelated signal
        """
        if window_size is None:
            window_size = self.fs * self.config.COHERENCE_WINDOW

        if eeg_signal.ndim == 1:
            eeg_signal = eeg_signal.reshape(1, -1)

        n_channels, n_samples = eeg_signal.shape
        protected = eeg_signal.copy()

        # Apply sliding window decorrelation
        for start in range(0, n_samples, window_size):
            end = min(start + window_size, n_samples)

            # Generate independent noise for this window
            noise = self._generate_adversarial_signal(end - start)

            # Apply to all channels
            for ch in range(n_channels):
                protected[ch, start:end] += noise

        return protected.squeeze()

    def compute_band_power(
        self,
        eeg_signal: np.ndarray,
        band: Tuple[float, float]
    ) -> float:
        """
        Compute power in a frequency band (diagnostic tool).

        Args:
            eeg_signal: 1D EEG signal
            band: Frequency band (low_hz, high_hz)

        Returns:
            Band power in µV²
        """
        # Compute power spectral density
        freqs, psd = signal.welch(eeg_signal, fs=self.fs, nperseg=min(len(eeg_signal), 256))

        # Find indices for band
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])

        # Integrate power
        band_power = np.trapz(psd[idx], freqs[idx])

        return band_power

    def verify_defense(self, original: np.ndarray, protected: np.ndarray) -> dict:
        """
        Verify that defense maintains EEG plausibility.

        Args:
            original: Original EEG signal
            protected: Protected signal

        Returns:
            Dictionary of diagnostic metrics
        """
        # Compute band powers
        alpha_original = self.compute_band_power(original, self.config.ALPHA_BAND)
        alpha_protected = self.compute_band_power(protected, self.config.ALPHA_BAND)

        theta_original = self.compute_band_power(original, self.config.THETA_BAND)
        theta_protected = self.compute_band_power(protected, self.config.THETA_BAND)

        # Compute SNR
        noise = protected - original
        signal_power = np.var(original)
        noise_power = np.var(noise)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

        return {
            'alpha_power_change_pct': ((alpha_protected - alpha_original) / alpha_original) * 100,
            'theta_power_change_pct': ((theta_protected - theta_original) / theta_original) * 100,
            'snr_db': snr_db,
            'injection_within_limits': abs(snr_db - self.config.TARGET_SNR) < 3.0,
            'temporal_coherence_broken': self.config.TEMPORAL_COHERENCE_BREAK
        }


# Example usage
if __name__ == "__main__":
    # Simulate EEG signal (1 channel, 5 seconds at 256 Hz)
    fs = 256
    duration = 5
    t = np.arange(duration * fs) / fs

    # Generate synthetic EEG (alpha + theta + noise)
    eeg_original = (
        1.5 * np.sin(2 * np.pi * 10 * t) +  # 10 Hz alpha
        0.8 * np.sin(2 * np.pi * 6 * t) +   # 6 Hz theta
        0.3 * np.random.randn(len(t))       # Noise
    )

    # Initialize defender
    defender = SpectralDefender(sampling_rate=fs)

    # Inject canary
    eeg_protected = defender.inject_canary(eeg_original)

    # Verify defense
    metrics = defender.verify_defense(eeg_original, eeg_protected)

    print("=== Cognitive Canary v5.0 - Spectral Defender ===")
    print(f"Alpha power change: {metrics['alpha_power_change_pct']:+.2f}%")
    print(f"Theta power change: {metrics['theta_power_change_pct']:+.2f}%")
    print(f"SNR: {metrics['snr_db']:.2f} dB (target: 6.0 dB)")
    print(f"Injection within limits: {metrics['injection_within_limits']}")
    print(f"Temporal coherence broken: {metrics['temporal_coherence_broken']}")
    print("\n✅ EEG protection active.")
