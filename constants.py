"""
Global Constants for Cognitive Canary
=====================================

Centralized configuration constants used across multiple modules.
Extracted from hardcoded values to improve maintainability and
enable easier parameter tuning.

Author: Cognitive Canary Project
License: MIT
Version: 6.0
"""

from typing import Tuple
import numpy as np


# =============================================================================
# SPECTRAL ANALYSIS CONSTANTS
# =============================================================================

# Target spectral entropy range for biomimicry (in nats)
# Natural human motor signals typically exhibit 3.0-3.5 nats
SPECTRAL_ENTROPY_MIN = 3.0  # Below this: too predictable/synthetic
SPECTRAL_ENTROPY_MAX = 3.5  # Above this: too random/unnatural
SPECTRAL_ENTROPY_TARGET = 3.2  # Optimal target for v6.0

# Default sampling frequency for behavioral signals (Hz)
DEFAULT_SAMPLING_FREQ = 100.0  # 100 Hz standard for cursor/keystroke

# FFT parameters
DEFAULT_FFT_NPERSEG = 256  # Welch's method segment length
DEFAULT_FFT_OVERLAP = 128  # 50% overlap for Welch's method


# =============================================================================
# FREQUENCY BAND DEFINITIONS (Hz)
# =============================================================================

# EEG frequency bands (standard neuroscience definitions)
BAND_DELTA: Tuple[float, float] = (0.5, 4.0)   # Deep sleep
BAND_THETA: Tuple[float, float] = (4.0, 8.0)   # Meditation, drowsiness
BAND_ALPHA: Tuple[float, float] = (8.0, 12.0)  # Relaxed wakefulness
BAND_BETA: Tuple[float, float] = (12.0, 30.0)  # Active thinking, focus
BAND_GAMMA: Tuple[float, float] = (30.0, 100.0)  # High-level cognition

# Tremor frequency bands (physiologic ranges)
TREMOR_BAND_PHYSIOLOGIC: Tuple[float, float] = (4.0, 12.0)  # Natural tremor
TREMOR_BAND_ESSENTIAL: Tuple[float, float] = (4.0, 12.0)    # Essential tremor
TREMOR_BAND_PARKINSONIAN: Tuple[float, float] = (3.0, 6.0)  # Parkinson's tremor

# Target injection band (overlaps with physiologic tremor for naturalism)
INJECTION_FREQ_BAND: Tuple[float, float] = (4.0, 12.0)


# =============================================================================
# LISSAJOUS CURVE PARAMETERS
# =============================================================================

# Frequency ratios (coprime integers for maximal ergodic coverage)
LISSAJOUS_FREQ_A = 13  # X-axis frequency ratio (prime)
LISSAJOUS_FREQ_B = 8   # Y-axis frequency ratio (coprime to A)
LISSAJOUS_FREQ_C = 5   # Z-axis frequency ratio (coprime to A, B)

# Phase offsets (radians)
LISSAJOUS_PHASE_X = np.pi / 2  # 90-degree offset for X-axis
LISSAJOUS_PHASE_Y = 0.0        # No offset for Y-axis
LISSAJOUS_PHASE_Z = np.pi / 4  # 45-degree offset for Z-axis

# Amplitude defaults (relative units)
LISSAJOUS_AMPLITUDE_DEFAULT = 5.0  # Base amplitude for 2D/3D curves

# Jitter parameters
LISSAJOUS_JITTER_PERCENTAGE = 2.0  # 2% physiologic jitter on amplitude


# =============================================================================
# KEYSTROKE JITTER PARAMETERS
# =============================================================================

# Temporal jitter (inter-key interval noise)
KEYSTROKE_TEMPORAL_JITTER_SIGMA = 0.012  # 12 ms standard deviation (pink noise)

# Pressure jitter (dwell time variability)
KEYSTROKE_PRESSURE_JITTER_SIGMA = 0.008  # 8 ms standard deviation (Gaussian)

# Directional jitter (typo injection)
KEYSTROKE_TYPO_PROBABILITY = 0.05  # 5% synthetic typo rate
KEYSTROKE_TYPO_BACKSPACE_DELAY = 0.15  # 150 ms cognitive delay before correction

# Typing speed constraints (WPM)
KEYSTROKE_WPM_MIN = 40  # Minimum plausible typing speed
KEYSTROKE_WPM_MAX = 80  # Maximum plausible typing speed (non-professional)


# =============================================================================
# ADAPTIVE TREMOR PARAMETERS
# =============================================================================

# Tremor extraction
TREMOR_DOMINANT_FREQ_MIN = 4.0   # Minimum detectable tremor frequency
TREMOR_DOMINANT_FREQ_MAX = 12.0  # Maximum detectable tremor frequency
TREMOR_BASELINE_SAMPLES = 500    # Minimum samples for calibration

# Calibration quality thresholds
TREMOR_SNR_MIN = 3.0  # Minimum SNR for reliable extraction (dB)
TREMOR_SNR_TARGET = 5.0  # Target SNR for good quality

# Injection parameters
TREMOR_INJECTION_SCALING = 1.0  # Default amplitude multiplier
TREMOR_PHASE_LOCK = True  # Maintain phase coherence by default


# =============================================================================
# SNR (SIGNAL-TO-NOISE RATIO) TARGETS
# =============================================================================

# Target SNR range for subtle injection (dB)
SNR_TARGET_MIN = 5.0  # Minimum acceptable SNR
SNR_TARGET_MAX = 7.0  # Maximum acceptable SNR (optimal: 6.0 dB)
SNR_TARGET_OPTIMAL = 6.0  # Sweet spot for biomimicry

# SNR enforcement
SNR_ENFORCEMENT_ENABLED = True  # Verify SNR in production


# =============================================================================
# TASK CLASSIFIER PARAMETERS
# =============================================================================

# Feature extraction windows
TASK_CLASSIFIER_WINDOW_SIZE = 5.0  # 5-second sliding window (seconds)

# Task detection thresholds (percentages as fractions)
TASK_CODING_SPECIAL_KEY_RATIO = 0.15      # 15% special keys (brackets, colons, etc.)
TASK_WRITING_KEYSTROKE_BURST_RATIO = 0.70  # 70% continuous typing bursts
TASK_WRITING_BACKSPACE_RATIO_MAX = 0.10    # <10% backspace rate
TASK_EMAIL_BURST_RATIO_MIN = 0.30          # 30% min burst ratio
TASK_EMAIL_BURST_RATIO_MAX = 0.60          # 60% max burst ratio
TASK_EMAIL_BACKSPACE_RATIO_MIN = 0.05      # >5% backspace (more editing than writing)
TASK_GAMING_CLICK_FREQ_MIN = 60            # 60 clicks/min minimum
TASK_DESIGN_DRAG_EVENTS_MIN = 10           # 10+ drag events in window

# Number of detectable task types
N_TASK_TYPES = 18  # v6.0 supports 18 task categories


# =============================================================================
# PRODUCTIVITY MONITORING PARAMETERS
# =============================================================================

# Productivity failsafe thresholds
PRODUCTIVITY_SLOWDOWN_THRESHOLD = 0.10  # 10% max allowed slowdown
PRODUCTIVITY_BASELINE_WINDOW = 1800.0   # 30 minutes baseline (seconds)
PRODUCTIVITY_MIN_CALIBRATION_TASKS = 10  # Minimum tasks for baseline calibration

# Auto-scaling parameters
PRODUCTIVITY_AUTOSCALE_STEP = 0.10  # Reduce injection by 10% per step
PRODUCTIVITY_INJECTION_MIN = 0.30   # Never reduce below 30% injection strength
PRODUCTIVITY_INJECTION_MAX = 1.00   # Maximum injection strength (100%)


# =============================================================================
# GRADIENT AUDITOR PARAMETERS
# =============================================================================

# Fingerprinting detection thresholds
GRADIENT_FINGERPRINT_TEMPORAL_STABILITY = 0.85  # 0.85 correlation = fingerprinting
GRADIENT_FINGERPRINT_MIN_SESSIONS = 3           # Need 3+ sessions for detection

# Gradient starvation detection
GRADIENT_STARVATION_ENTROPY_THRESHOLD = 0.5  # Entropy < 0.5 = starvation
GRADIENT_STARVATION_CONCENTRATION_TOP_K = 0.2  # Top 20% of features

# Re-identification detection
GRADIENT_REIDENT_CORRELATION_THRESHOLD = 0.85  # Cross-session correlation

# Federated learning poisoning detection (v6.0)
GRADIENT_FL_DIVERGENCE_THRESHOLD = 0.02  # |∇w(t) - ∇w(t-1)| > 0.02
GRADIENT_FL_BYZANTINE_NORM_THRESHOLD = 5.0  # ||∇w|| > 5.0 = Byzantine attack
GRADIENT_FL_MIN_ROUNDS = 5  # Need 5+ FL rounds for baseline


# =============================================================================
# 3D OBFUSCATION PARAMETERS
# =============================================================================

# Z-axis discretization (scroll/zoom events)
Z_AXIS_SCROLL_PROBABILITY = 0.70  # 70% of Z-axis maps to scroll
Z_AXIS_ZOOM_PROBABILITY = 0.30    # 30% of Z-axis maps to zoom

# Scroll/zoom magnitude scaling
Z_AXIS_SCROLL_SCALE = 10.0  # Pixels per scroll unit
Z_AXIS_ZOOM_SCALE = 0.01    # Zoom delta per unit (1% zoom steps)

# 3D coverage validation
MIN_3D_COVERAGE = 0.70  # Minimum volume coverage (70%)


# =============================================================================
# SPECTRAL CANARY (EEG) PARAMETERS
# =============================================================================

# Injection frequency mixture
EEG_ALPHA_WEIGHT = 0.60  # 60% alpha band (8-12 Hz)
EEG_THETA_WEIGHT = 0.40  # 40% theta band (4-8 Hz)

# Temporal coherence breaking
EEG_COHERENCE_WINDOW_SIZE = 5.0  # 5-second sliding window (seconds)

# Phase randomization
EEG_PHASE_RANDOMIZATION = True  # Enable phase randomization by default


# =============================================================================
# INTEGRATION & DEPLOYMENT PARAMETERS
# =============================================================================

# Injection profiles (from task_modulator.json)
PROFILE_STEALTH_INJECTION_PROB = 0.30       # 30% injection in stealth mode
PROFILE_BALANCED_INJECTION_PROB = 0.70      # 70% injection in balanced mode (default)
PROFILE_MAXIMUM_INJECTION_PROB = 1.00       # 100% injection in maximum defense mode
PROFILE_RESEARCH_INJECTION_PROB = 0.00      # 0% injection in research mode (logging only)

# Latency constraints
MAX_LATENCY_MS = 50.0  # Maximum acceptable latency (milliseconds)

# Logging & monitoring
ENABLE_PERFORMANCE_LOGGING = True  # Log performance metrics by default
ENABLE_ATTACK_LOGGING = True       # Log detected attacks by default


# =============================================================================
# VERSION & METADATA
# =============================================================================

VERSION = "6.0"
VERSION_MAJOR = 6
VERSION_MINOR = 0
VERSION_PATCH = 0

PROJECT_NAME = "Cognitive Canary"
PROJECT_DESCRIPTION = "Active Defense Against Neural Inference"

# License
LICENSE = "MIT"

# Citation
CITATION_YEAR = 2025
CITATION_AUTHORS = "Cognitive Canary Project Contributors"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_injection_profile(profile_name: str) -> dict:
    """
    Get injection profile configuration by name.

    Parameters
    ----------
    profile_name : str
        One of: 'stealth', 'balanced', 'maximum', 'research'

    Returns
    -------
    dict
        Profile configuration with injection_probability and other settings
    """
    profiles = {
        'stealth': {
            'injection_probability': PROFILE_STEALTH_INJECTION_PROB,
            'gradient_audit_sensitivity': 'low',
            'productivity_enforcement': True,
        },
        'balanced': {
            'injection_probability': PROFILE_BALANCED_INJECTION_PROB,
            'gradient_audit_sensitivity': 'medium',
            'productivity_enforcement': True,
        },
        'maximum': {
            'injection_probability': PROFILE_MAXIMUM_INJECTION_PROB,
            'gradient_audit_sensitivity': 'high',
            'productivity_enforcement': True,
        },
        'research': {
            'injection_probability': PROFILE_RESEARCH_INJECTION_PROB,
            'gradient_audit_sensitivity': 'medium',
            'productivity_enforcement': False,
        }
    }

    if profile_name not in profiles:
        raise ValueError(f"Unknown profile '{profile_name}'. Choose from: {list(profiles.keys())}")

    return profiles[profile_name]


def validate_constants() -> bool:
    """
    Validate that all constants are within acceptable ranges.

    Returns
    -------
    bool
        True if all constants are valid

    Raises
    ------
    ValueError
        If any constant is invalid
    """
    # Validate frequency bands
    assert BAND_THETA[0] < BAND_THETA[1], "Invalid BAND_THETA range"
    assert BAND_ALPHA[0] < BAND_ALPHA[1], "Invalid BAND_ALPHA range"
    assert TREMOR_BAND_PHYSIOLOGIC[0] < TREMOR_BAND_PHYSIOLOGIC[1], "Invalid TREMOR_BAND"

    # Validate Lissajous parameters
    assert LISSAJOUS_FREQ_A > 0, "LISSAJOUS_FREQ_A must be positive"
    assert LISSAJOUS_FREQ_B > 0, "LISSAJOUS_FREQ_B must be positive"
    assert LISSAJOUS_FREQ_C > 0, "LISSAJOUS_FREQ_C must be positive"
    assert np.gcd(LISSAJOUS_FREQ_A, LISSAJOUS_FREQ_B) == 1, "A and B must be coprime"

    # Validate thresholds
    assert 0.0 < PRODUCTIVITY_SLOWDOWN_THRESHOLD < 1.0, "Invalid productivity threshold"
    assert 0.0 <= PROFILE_STEALTH_INJECTION_PROB <= 1.0, "Invalid injection probability"

    print("✓ All constants validated successfully")
    return True


if __name__ == "__main__":
    """
    Example usage and validation.
    """
    print("=" * 60)
    print(f"{PROJECT_NAME} v{VERSION} - Global Constants")
    print("=" * 60)

    print(f"\n📊 Spectral Parameters:")
    print(f"   Target Entropy: {SPECTRAL_ENTROPY_TARGET} nats")
    print(f"   Tremor Band: {TREMOR_BAND_PHYSIOLOGIC[0]}-{TREMOR_BAND_PHYSIOLOGIC[1]} Hz")
    print(f"   SNR Target: {SNR_TARGET_OPTIMAL} dB")

    print(f"\n🎯 Lissajous Parameters:")
    print(f"   Frequency Ratios: {LISSAJOUS_FREQ_A}:{LISSAJOUS_FREQ_B}:{LISSAJOUS_FREQ_C}")
    print(f"   Physiologic Jitter: {LISSAJOUS_JITTER_PERCENTAGE}%")

    print(f"\n⌨️  Keystroke Parameters:")
    print(f"   Temporal Jitter: {KEYSTROKE_TEMPORAL_JITTER_SIGMA*1000:.1f} ms")
    print(f"   Typo Rate: {KEYSTROKE_TYPO_PROBABILITY*100:.1f}%")

    print(f"\n🛡️  Gradient Auditor:")
    print(f"   Fingerprint Threshold: {GRADIENT_FINGERPRINT_TEMPORAL_STABILITY}")
    print(f"   FL Poisoning Detection: ε={GRADIENT_FL_DIVERGENCE_THRESHOLD}")

    print(f"\n⚙️  Injection Profiles:")
    for profile_name in ['stealth', 'balanced', 'maximum', 'research']:
        profile = get_injection_profile(profile_name)
        print(f"   {profile_name.capitalize()}: {profile['injection_probability']*100:.0f}% injection")

    print(f"\n✅ Validation:")
    validate_constants()

    print("\n" + "=" * 60)
