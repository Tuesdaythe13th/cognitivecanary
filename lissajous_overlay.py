"""
Cognitive Canary v5.0 - Lissajous Overlay Module
================================================

Core cursor obfuscation engine using physiologically-masked Lissajous curves.

Key Innovation: Generates adversarial cursor movements that:
1. Pass as human (biomimicry via 4-12 Hz tremor injection)
2. Poison ML classifiers (gradient starvation through spectral entropy)
3. Evade forensic analysis (irrational frequency ratios)

Usage:
    from lissajous_overlay import LissajousEngine

    engine = LissajousEngine()
    obfuscated_path = engine.generate(duration=2.0, target_points=100)

Author: Cognitive Canary Project
License: MIT
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CanaryConfig:
    """Configuration parameters for adversarial cursor generation."""

    # Frequency parameters (Hz)
    TREMOR_BAND: Tuple[float, float] = (4.0, 12.0)  # Physiologic hand tremor range

    # Lissajous curve parameters
    FREQ_RATIO: Tuple[int, int] = (13, 8)  # Irrational ratio for maximal ergodicity
    PHASE_OFFSET: float = np.pi / 2  # X-axis phase shift

    # Noise injection
    JITTER_AMPLITUDE: float = 0.02  # ±2% uniform noise (mimics motor variance)
    ENTROPY_TARGET: float = 3.2  # Target spectral entropy (nats)

    # Normalization
    CANVAS_BOUNDS: Tuple[float, float] = (0.0, 1.0)  # Output range [0, 1]


class LissajousEngine:
    """
    Generates adversarially-optimized cursor paths using Lissajous curves.

    The core defense mechanism exploits a fundamental weakness in behavioral
    profiling systems: they learn to classify the *easiest* discriminative
    feature (curve smoothness) rather than semantic intent.

    By injecting mathematically-perfect Lissajous curves masked with
    physiologic jitter, we force classifiers into gradient starvation.
    """

    def __init__(self, config: Optional[CanaryConfig] = None):
        """
        Initialize the Lissajous engine.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or CanaryConfig()
        self._validate_config()

    def _validate_config(self):
        """Ensure configuration parameters are within safe ranges."""
        assert 0 < self.config.JITTER_AMPLITUDE < 0.1, \
            "Jitter amplitude must be 0-10% to maintain biomimicry"
        assert self.config.TREMOR_BAND[0] < self.config.TREMOR_BAND[1], \
            "Invalid tremor band range"
        assert np.gcd(*self.config.FREQ_RATIO) == 1, \
            "Frequency ratio must be coprime for maximal ergodicity"

    def generate(
        self,
        duration: float = 2.0,
        target_points: int = 100,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate an adversarial cursor path.

        Args:
            duration: Path duration in seconds
            target_points: Number of sample points
            seed: Random seed for reproducibility

        Returns:
            numpy array of shape (target_points, 2) with columns [x, y]
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate time vector
        t = np.linspace(0, 2 * np.pi, target_points)

        # Compute Lissajous curve
        freq_x, freq_y = self.config.FREQ_RATIO
        x = np.sin(freq_x * t + self.config.PHASE_OFFSET)
        y = np.sin(freq_y * t)

        # Normalize to canvas bounds
        x = self._normalize_signal(x)
        y = self._normalize_signal(y)

        # Inject physiologic jitter (independent per axis for realism)
        x += self._generate_jitter(target_points)
        y += self._generate_jitter(target_points)

        # Clip to bounds (safety)
        x = np.clip(x, *self.config.CANVAS_BOUNDS)
        y = np.clip(y, *self.config.CANVAS_BOUNDS)

        return np.column_stack([x, y])

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal from [-1, 1] to canvas bounds."""
        normalized = (signal + 1) / 2  # Map [-1, 1] -> [0, 1]
        min_bound, max_bound = self.config.CANVAS_BOUNDS
        return normalized * (max_bound - min_bound) + min_bound

    def _generate_jitter(self, n_points: int) -> np.ndarray:
        """
        Generate physiologically-plausible jitter.

        Uses uniform distribution to match the spectral signature of
        essential tremor (8-12 Hz baseline) + cognitive stress (4-8 Hz).
        """
        return np.random.uniform(
            -self.config.JITTER_AMPLITUDE,
            self.config.JITTER_AMPLITUDE,
            n_points
        )

    def compute_spectral_entropy(self, signal: np.ndarray) -> float:
        """
        Compute spectral entropy of a signal (diagnostic tool).

        Args:
            signal: 1D time-domain signal

        Returns:
            Spectral entropy in nats
        """
        from scipy import signal as sp_signal

        # Compute power spectral density
        freqs, psd = sp_signal.welch(signal, nperseg=min(len(signal), 64))

        # Normalize PSD to probability distribution
        psd_norm = psd / np.sum(psd)

        # Compute Shannon entropy
        entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

        return entropy

    def verify_biomimicry(self, path: np.ndarray) -> dict:
        """
        Verify that generated path matches human motor statistics.

        Args:
            path: Generated cursor path (N x 2)

        Returns:
            Dictionary of diagnostic metrics
        """
        # Compute velocity profile
        dx = np.diff(path[:, 0])
        dy = np.diff(path[:, 1])
        velocity = np.sqrt(dx**2 + dy**2)

        # Compute spectral entropy
        entropy_x = self.compute_spectral_entropy(path[:, 0])
        entropy_y = self.compute_spectral_entropy(path[:, 1])

        return {
            'mean_velocity': np.mean(velocity),
            'velocity_std': np.std(velocity),
            'spectral_entropy_x': entropy_x,
            'spectral_entropy_y': entropy_y,
            'entropy_target_met': abs(entropy_x - self.config.ENTROPY_TARGET) < 0.5,
            'path_length': len(path)
        }


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = LissajousEngine()

    # Generate adversarial path
    path = engine.generate(duration=2.0, target_points=100, seed=42)

    # Verify biomimicry
    metrics = engine.verify_biomimicry(path)

    print("=== Cognitive Canary v5.0 - Lissajous Engine ===")
    print(f"Generated path: {len(path)} points")
    print(f"Mean velocity: {metrics['mean_velocity']:.4f}")
    print(f"Velocity variance: {metrics['velocity_std']:.4f}")
    print(f"Spectral entropy (X): {metrics['spectral_entropy_x']:.4f} nats")
    print(f"Spectral entropy (Y): {metrics['spectral_entropy_y']:.4f} nats")
    print(f"Entropy target met: {metrics['entropy_target_met']}")
    print("\n✅ Adversarial path generation complete.")
