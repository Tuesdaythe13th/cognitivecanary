"""
Cognitive Canary v5.0 - Lissajous Overlay Module (DEPRECATED)
=============================================================

⚠️ DEPRECATED: This module is v5.0 legacy code.
Use lissajous_3d.py instead - it supports both 2D and 3D obfuscation.
This file is kept for backward compatibility only.

Core cursor obfuscation engine using physiologically-masked Lissajous curves.

Key Innovation: Generates adversarial cursor movements that:
1. Pass as human (biomimicry via 4-12 Hz tremor injection)
2. Poison ML classifiers (gradient starvation through spectral entropy)
3. Evade forensic analysis (irrational frequency ratios)

Usage (DEPRECATED):
    from lissajous_overlay import LissajousEngine  # Use lissajous_3d.Lissajous3DEngine instead

    engine = LissajousEngine()
    obfuscated_path = engine.generate(duration=2.0, target_points=100)

Author: Cognitive Canary Project
License: MIT
"""

import warnings
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

# Import shared utilities (v6.0)
from spectral_utils import compute_spectral_entropy, normalize_signal
from noise_generators import generate_uniform_jitter
from constants import (
    TREMOR_BAND_PHYSIOLOGIC,
    LISSAJOUS_FREQ_A, LISSAJOUS_FREQ_B, LISSAJOUS_PHASE_X,
    LISSAJOUS_JITTER_PERCENTAGE,
    SPECTRAL_ENTROPY_TARGET
)


@dataclass
class CanaryConfig:
    """Configuration parameters for adversarial cursor generation."""

    # Frequency parameters (Hz) - imported from constants.py
    TREMOR_BAND: Tuple[float, float] = TREMOR_BAND_PHYSIOLOGIC  # Physiologic hand tremor range (4-12 Hz)

    # Lissajous curve parameters - imported from constants.py
    FREQ_RATIO: Tuple[int, int] = (LISSAJOUS_FREQ_A, LISSAJOUS_FREQ_B)  # Coprime ratio (13, 8) for maximal ergodicity
    PHASE_OFFSET: float = LISSAJOUS_PHASE_X  # X-axis phase shift (π/2)

    # Noise injection - imported from constants.py
    JITTER_AMPLITUDE: float = LISSAJOUS_JITTER_PERCENTAGE / 100.0  # ±2% uniform noise (mimics motor variance)
    ENTROPY_TARGET: float = SPECTRAL_ENTROPY_TARGET  # Target spectral entropy (3.2 nats)

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

        ⚠️ DEPRECATED: Use lissajous_3d.Lissajous3DEngine instead.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        warnings.warn(
            "lissajous_overlay.LissajousEngine is deprecated. "
            "Use lissajous_3d.Lissajous3DEngine instead for both 2D and 3D obfuscation.",
            DeprecationWarning,
            stacklevel=2
        )
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

        # Normalize to canvas bounds using shared utility
        x = normalize_signal(x, target_range=self.config.CANVAS_BOUNDS)
        y = normalize_signal(y, target_range=self.config.CANVAS_BOUNDS)

        # Inject physiologic jitter (independent per axis for realism)
        x += self._generate_jitter(target_points)
        y += self._generate_jitter(target_points)

        # Clip to bounds (safety)
        x = np.clip(x, *self.config.CANVAS_BOUNDS)
        y = np.clip(y, *self.config.CANVAS_BOUNDS)

        return np.column_stack([x, y])

    def _generate_jitter(self, n_points: int) -> np.ndarray:
        """
        Generate physiologically-plausible jitter using shared utility.

        Uses uniform distribution to match the spectral signature of
        essential tremor (8-12 Hz baseline) + cognitive stress (4-8 Hz).
        """
        # Use shared noise generator (peak-to-peak amplitude = 2 * jitter_amplitude)
        return generate_uniform_jitter(n_points, amplitude=2 * self.config.JITTER_AMPLITUDE)

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

        # Compute spectral entropy using shared utility
        entropy_x = compute_spectral_entropy(path[:, 0], fs=100.0)
        entropy_y = compute_spectral_entropy(path[:, 1], fs=100.0)

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
