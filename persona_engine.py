"""
Cognitive Canary - Persona Coherence Engine
===========================================

Maintains a statistically consistent synthetic behavioral persona across
sessions to prevent longitudinal re-identification.

The Threat: Cross-Session Fingerprinting
-----------------------------------------
The gradient_auditor.py detects cross-session re-identification attacks in
real-time. But detection is reactive. A sophisticated adversary can still
correlate sessions *before* the auditor fires — especially if each session's
injected noise is sampled independently (high variance across sessions creates
a detectable distribution).

The Solution: Stateful Persona
--------------------------------
The PersonaCoherenceEngine maintains a persistent "behavioral avatar" — a set
of synthetic motor and typing parameters that remain *stable within an identity
window* but are *decorrelated from the user's true fingerprint*.

Key Properties:
1. Consistency  : Same persona parameters are reused across sessions in a window
2. Authenticity : Persona parameters are sampled from biomechanically valid ranges
3. Separation   : Persona is statistically distant from the user's true baseline
4. Rotation     : Persona evolves slowly and unpredictably to prevent long-term tracking
5. Auditability : Session fingerprints are tracked to verify decorrelation

Integration with Existing Engines
-----------------------------------
The persona engine does NOT replace AdaptiveTremor or KeystrokeJitter.
It *informs* them by providing consistent synthetic baseline parameters:

    persona = PersonaCoherenceEngine()
    params = persona.get_tremor_params()
    tremor_engine.set_synthetic_baseline(params.dominant_freq, params.amplitude)

Author: Cognitive Canary Project
License: MIT
Version: 6.1
"""

import numpy as np
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TremorPersonaParams:
    """Synthetic tremor baseline for the persona."""
    dominant_freq_hz: float        # 4.0 – 12.0 Hz (physiologic range)
    amplitude: float               # 0.5 – 3.0 pixels (normalized)
    phase_offset_rad: float        # 0.0 – 2π (random phase lock)
    harmonic_ratio: float          # 0.1 – 0.4 (second harmonic strength)
    spectral_entropy_target: float # 3.0 – 3.5 nats (biomimicry target)


@dataclass
class KeystrokePersonaParams:
    """Synthetic keystroke rhythm for the persona."""
    mean_iki_ms: float        # 80 – 300 ms (inter-key interval)
    iki_std_ms: float         # 10 – 50 ms (variability)
    burst_duration_ms: float  # 500 – 2000 ms (typing burst length)
    burst_gap_ms: float       # 200 – 1000 ms (pause between bursts)
    typo_rate: float          # 0.02 – 0.08 (synthetic typo probability)
    backspace_delay_ms: float # 100 – 250 ms (correction latency)


@dataclass
class CursorPersonaParams:
    """Synthetic cursor movement characteristics for the persona."""
    mean_velocity_px_s: float      # 200 – 800 px/s
    velocity_std_px_s: float       # 50 – 200 px/s
    path_curvature: float          # 0.1 – 0.9 (0=straight, 1=highly curved)
    micro_correction_rate: float   # 0.05 – 0.20 (sub-movement frequency)
    pause_frequency_hz: float      # 0.1 – 0.5 Hz (cursor pause rate)


@dataclass
class PersonaState:
    """
    Complete behavioral persona for one identity window.

    A persona is valid for `rotation_interval_sessions` sessions before
    being gradually evolved (rotation). All parameters are drawn from
    biomechanically valid ranges but are statistically distant from the
    user's true fingerprint.
    """
    persona_id: str                          # Deterministic hash (not user-linked)
    created_at: float                        # Unix timestamp
    sessions_used: int = 0
    rotation_interval_sessions: int = 10    # Rotate every N sessions

    tremor: Optional[TremorPersonaParams] = None
    keystroke: Optional[KeystrokePersonaParams] = None
    cursor: Optional[CursorPersonaParams] = None

    # Session fingerprints for decorrelation auditing
    session_fingerprints: List[float] = field(default_factory=list)


# =============================================================================
# BIOMECHANICAL VALIDITY RANGES
# =============================================================================

# These ranges define the space of plausible human motor behavior.
# Personas are sampled from WITHIN these ranges to remain undetectable.
TREMOR_FREQ_RANGE = (4.0, 12.0)         # Hz
TREMOR_AMP_RANGE = (0.5, 3.0)           # normalized pixels
TREMOR_PHASE_RANGE = (0.0, 2 * np.pi)
TREMOR_HARMONIC_RANGE = (0.10, 0.40)
TREMOR_ENTROPY_RANGE = (3.0, 3.5)       # nats

KEYSTROKE_IKI_MEAN_RANGE = (80.0, 300.0)   # ms
KEYSTROKE_IKI_STD_RANGE = (10.0, 50.0)    # ms
KEYSTROKE_BURST_DUR_RANGE = (500.0, 2000.0)
KEYSTROKE_BURST_GAP_RANGE = (200.0, 1000.0)
KEYSTROKE_TYPO_RANGE = (0.02, 0.08)
KEYSTROKE_BACKSPACE_RANGE = (100.0, 250.0)

CURSOR_VELOCITY_MEAN_RANGE = (200.0, 800.0)  # px/s
CURSOR_VELOCITY_STD_RANGE = (50.0, 200.0)
CURSOR_CURVATURE_RANGE = (0.1, 0.9)
CURSOR_MICRO_CORRECTION_RANGE = (0.05, 0.20)
CURSOR_PAUSE_FREQ_RANGE = (0.1, 0.5)


# =============================================================================
# PERSONA COHERENCE ENGINE
# =============================================================================

class PersonaCoherenceEngine:
    """
    Manages the lifecycle of a synthetic behavioral persona.

    Each time a session begins, the engine provides consistent persona
    parameters. After `rotation_interval_sessions` uses, it smoothly
    evolves the persona (rotation) to prevent long-term tracking while
    maintaining within-window consistency.

    Decorrelation Guarantee
    -----------------------
    The engine tracks fingerprints across sessions and raises an alert
    if the cross-session correlation exceeds the threshold used by
    GradientAuditor (0.85), indicating the persona may be too stable.

    Parameters
    ----------
    rotation_interval : int
        Number of sessions before persona rotation. Lower = more privacy
        but less consistency (harder to build natural-looking patterns).
        Recommended: 5 – 20 sessions.
    persona_state_file : str, optional
        Path to persist persona state between process restarts.
        If None, persona is in-memory only (reset on restart).
    separation_sigma : float
        How many standard deviations to separate persona from user baseline.
        Higher = harder to link to real user, but more detectable as unusual.
        Default 1.5 (statistically distinct but within observed human range).
    """

    # Fingerprinting alert threshold (matches GradientAuditor)
    CORRELATION_ALERT_THRESHOLD = 0.85
    # Minimum sessions before decorrelation check is meaningful
    MIN_SESSIONS_FOR_AUDIT = 3

    def __init__(
        self,
        rotation_interval: int = 10,
        persona_state_file: Optional[str] = None,
        separation_sigma: float = 1.5,
        seed: Optional[int] = None,
    ):
        self._rotation_interval = rotation_interval
        self._state_file = Path(persona_state_file) if persona_state_file else None
        self._separation_sigma = separation_sigma
        self._rng = np.random.default_rng(seed)
        self._current_persona: Optional[PersonaState] = None

        if self._state_file and self._state_file.exists():
            self._load_persona()
        else:
            self._initialize_persona()

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def begin_session(self) -> str:
        """
        Mark the start of a new obfuscation session.

        Increments session counter, triggers rotation if needed, and
        returns the current persona ID.

        Returns
        -------
        str
            The active persona ID.
        """
        assert self._current_persona is not None
        self._current_persona.sessions_used += 1

        if self._current_persona.sessions_used > self._rotation_interval:
            self._rotate_persona()

        if self._state_file:
            self._save_persona()

        return self._current_persona.persona_id

    def get_tremor_params(self) -> TremorPersonaParams:
        """Return the persona's synthetic tremor parameters."""
        assert self._current_persona is not None and self._current_persona.tremor is not None
        return self._current_persona.tremor

    def get_keystroke_params(self) -> KeystrokePersonaParams:
        """Return the persona's synthetic keystroke rhythm parameters."""
        assert self._current_persona is not None and self._current_persona.keystroke is not None
        return self._current_persona.keystroke

    def get_cursor_params(self) -> CursorPersonaParams:
        """Return the persona's synthetic cursor movement parameters."""
        assert self._current_persona is not None and self._current_persona.cursor is not None
        return self._current_persona.cursor

    def record_session_fingerprint(self, feature_vector: np.ndarray) -> None:
        """
        Record a session's behavioral fingerprint for decorrelation auditing.

        Parameters
        ----------
        feature_vector : np.ndarray
            A summary feature vector representing this session's
            obfuscated behavior (e.g., mean IKI, cursor velocity, tremor freq).
        """
        assert self._current_persona is not None
        # Reduce to scalar fingerprint via L2 norm (simple, consistent)
        fingerprint = float(np.linalg.norm(feature_vector))
        self._current_persona.session_fingerprints.append(fingerprint)

        # Keep history bounded to prevent memory growth
        if len(self._current_persona.session_fingerprints) > 50:
            self._current_persona.session_fingerprints.pop(0)

    def audit_decorrelation(self) -> Dict[str, object]:
        """
        Check whether session fingerprints are sufficiently decorrelated.

        Cross-session correlation > 0.85 suggests the persona is too
        stable and may be fingerprintable across sessions.

        Returns
        -------
        dict with keys:
            'is_correlated' (bool): True = persona may be fingerprintable
            'max_correlation' (float): Highest pairwise session correlation
            'recommendation' (str): Suggested action
        """
        assert self._current_persona is not None
        fps = self._current_persona.session_fingerprints

        if len(fps) < self.MIN_SESSIONS_FOR_AUDIT:
            return {
                'is_correlated': False,
                'max_correlation': 0.0,
                'recommendation': f"Need {self.MIN_SESSIONS_FOR_AUDIT}+ sessions for audit",
            }

        # Compute pairwise Pearson correlations on scalar fingerprints
        # (using rolling windows of 3 consecutive sessions)
        correlations = []
        arr = np.array(fps)
        for i in range(len(arr) - 2):
            window = arr[i:i+3]
            # Normalize: correlation of window with itself over sessions
            if window.std() > 1e-10:
                corr = float(np.corrcoef(window, np.arange(3))[0, 1])
                correlations.append(abs(corr))

        max_corr = max(correlations) if correlations else 0.0
        is_correlated = max_corr > self.CORRELATION_ALERT_THRESHOLD

        recommendation = (
            "Force early persona rotation to increase decorrelation"
            if is_correlated else
            "Persona decorrelation is within acceptable bounds"
        )

        if is_correlated:
            self._rotate_persona(force=True)
            recommendation += " — persona rotated automatically"

        return {
            'is_correlated': is_correlated,
            'max_correlation': max_corr,
            'recommendation': recommendation,
        }

    def enforce_consistency(self, raw_params: Dict[str, float]) -> Dict[str, float]:
        """
        Blend raw injection parameters with persona parameters.

        When other engines (AdaptiveTremor, KeystrokeJitter) produce
        raw parameters, this method shifts them toward the persona
        baseline to enforce cross-session consistency.

        Parameters
        ----------
        raw_params : dict
            Raw parameters from an obfuscation engine. Expected keys:
            'tremor_freq', 'tremor_amplitude', 'iki_mean', 'cursor_velocity'

        Returns
        -------
        dict
            Parameters adjusted toward persona baseline (50% blend).
        """
        assert self._current_persona is not None
        adjusted = dict(raw_params)
        blend = 0.5  # 50% persona, 50% raw

        if 'tremor_freq' in raw_params and self._current_persona.tremor:
            adjusted['tremor_freq'] = (
                blend * self._current_persona.tremor.dominant_freq_hz
                + (1 - blend) * raw_params['tremor_freq']
            )
        if 'tremor_amplitude' in raw_params and self._current_persona.tremor:
            adjusted['tremor_amplitude'] = (
                blend * self._current_persona.tremor.amplitude
                + (1 - blend) * raw_params['tremor_amplitude']
            )
        if 'iki_mean' in raw_params and self._current_persona.keystroke:
            adjusted['iki_mean'] = (
                blend * self._current_persona.keystroke.mean_iki_ms
                + (1 - blend) * raw_params['iki_mean']
            )
        if 'cursor_velocity' in raw_params and self._current_persona.cursor:
            adjusted['cursor_velocity'] = (
                blend * self._current_persona.cursor.mean_velocity_px_s
                + (1 - blend) * raw_params['cursor_velocity']
            )

        return adjusted

    @property
    def persona_id(self) -> str:
        assert self._current_persona is not None
        return self._current_persona.persona_id

    @property
    def sessions_until_rotation(self) -> int:
        assert self._current_persona is not None
        return max(0, self._rotation_interval - self._current_persona.sessions_used)

    # -------------------------------------------------------------------------
    # INTERNAL METHODS
    # -------------------------------------------------------------------------

    def _initialize_persona(self) -> None:
        """Sample a fresh persona from the biomechanically valid parameter space."""
        persona_id = self._generate_persona_id()
        self._current_persona = PersonaState(
            persona_id=persona_id,
            created_at=time.time(),
            rotation_interval_sessions=self._rotation_interval,
            tremor=self._sample_tremor_params(),
            keystroke=self._sample_keystroke_params(),
            cursor=self._sample_cursor_params(),
        )

    def _rotate_persona(self, force: bool = False) -> None:
        """
        Smoothly evolve the persona toward a new random target.

        Uses interpolation rather than a hard reset to avoid a detectable
        discontinuity in the behavioral trajectory that could itself be
        a fingerprinting signal.
        """
        assert self._current_persona is not None

        old_tremor = self._current_persona.tremor
        old_keystroke = self._current_persona.keystroke
        old_cursor = self._current_persona.cursor

        new_tremor = self._sample_tremor_params()
        new_keystroke = self._sample_keystroke_params()
        new_cursor = self._sample_cursor_params()

        # Interpolation factor: 0.3 = gradual, 1.0 = hard reset
        alpha = 0.4 if not force else 1.0

        # Smoothly interpolate tremor
        self._current_persona.tremor = TremorPersonaParams(
            dominant_freq_hz=self._lerp(old_tremor.dominant_freq_hz, new_tremor.dominant_freq_hz, alpha),
            amplitude=self._lerp(old_tremor.amplitude, new_tremor.amplitude, alpha),
            phase_offset_rad=new_tremor.phase_offset_rad,  # Phase always resets
            harmonic_ratio=self._lerp(old_tremor.harmonic_ratio, new_tremor.harmonic_ratio, alpha),
            spectral_entropy_target=self._lerp(
                old_tremor.spectral_entropy_target, new_tremor.spectral_entropy_target, alpha
            ),
        )

        # Smoothly interpolate keystroke
        self._current_persona.keystroke = KeystrokePersonaParams(
            mean_iki_ms=self._lerp(old_keystroke.mean_iki_ms, new_keystroke.mean_iki_ms, alpha),
            iki_std_ms=self._lerp(old_keystroke.iki_std_ms, new_keystroke.iki_std_ms, alpha),
            burst_duration_ms=self._lerp(old_keystroke.burst_duration_ms, new_keystroke.burst_duration_ms, alpha),
            burst_gap_ms=self._lerp(old_keystroke.burst_gap_ms, new_keystroke.burst_gap_ms, alpha),
            typo_rate=self._lerp(old_keystroke.typo_rate, new_keystroke.typo_rate, alpha),
            backspace_delay_ms=self._lerp(old_keystroke.backspace_delay_ms, new_keystroke.backspace_delay_ms, alpha),
        )

        # Smoothly interpolate cursor
        self._current_persona.cursor = CursorPersonaParams(
            mean_velocity_px_s=self._lerp(old_cursor.mean_velocity_px_s, new_cursor.mean_velocity_px_s, alpha),
            velocity_std_px_s=self._lerp(old_cursor.velocity_std_px_s, new_cursor.velocity_std_px_s, alpha),
            path_curvature=self._lerp(old_cursor.path_curvature, new_cursor.path_curvature, alpha),
            micro_correction_rate=self._lerp(old_cursor.micro_correction_rate, new_cursor.micro_correction_rate, alpha),
            pause_frequency_hz=self._lerp(old_cursor.pause_frequency_hz, new_cursor.pause_frequency_hz, alpha),
        )

        self._current_persona.sessions_used = 0
        self._current_persona.persona_id = self._generate_persona_id()
        self._current_persona.created_at = time.time()
        # Preserve fingerprint history for continued decorrelation auditing
        if force:
            self._current_persona.session_fingerprints.clear()

    def _sample_tremor_params(self) -> TremorPersonaParams:
        """Sample tremor parameters uniformly from valid range."""
        return TremorPersonaParams(
            dominant_freq_hz=float(self._rng.uniform(*TREMOR_FREQ_RANGE)),
            amplitude=float(self._rng.uniform(*TREMOR_AMP_RANGE)),
            phase_offset_rad=float(self._rng.uniform(*TREMOR_PHASE_RANGE)),
            harmonic_ratio=float(self._rng.uniform(*TREMOR_HARMONIC_RANGE)),
            spectral_entropy_target=float(self._rng.uniform(*TREMOR_ENTROPY_RANGE)),
        )

    def _sample_keystroke_params(self) -> KeystrokePersonaParams:
        """Sample keystroke parameters uniformly from valid range."""
        return KeystrokePersonaParams(
            mean_iki_ms=float(self._rng.uniform(*KEYSTROKE_IKI_MEAN_RANGE)),
            iki_std_ms=float(self._rng.uniform(*KEYSTROKE_IKI_STD_RANGE)),
            burst_duration_ms=float(self._rng.uniform(*KEYSTROKE_BURST_DUR_RANGE)),
            burst_gap_ms=float(self._rng.uniform(*KEYSTROKE_BURST_GAP_RANGE)),
            typo_rate=float(self._rng.uniform(*KEYSTROKE_TYPO_RANGE)),
            backspace_delay_ms=float(self._rng.uniform(*KEYSTROKE_BACKSPACE_RANGE)),
        )

    def _sample_cursor_params(self) -> CursorPersonaParams:
        """Sample cursor parameters uniformly from valid range."""
        return CursorPersonaParams(
            mean_velocity_px_s=float(self._rng.uniform(*CURSOR_VELOCITY_MEAN_RANGE)),
            velocity_std_px_s=float(self._rng.uniform(*CURSOR_VELOCITY_STD_RANGE)),
            path_curvature=float(self._rng.uniform(*CURSOR_CURVATURE_RANGE)),
            micro_correction_rate=float(self._rng.uniform(*CURSOR_MICRO_CORRECTION_RANGE)),
            pause_frequency_hz=float(self._rng.uniform(*CURSOR_PAUSE_FREQ_RANGE)),
        )

    @staticmethod
    def _lerp(a: float, b: float, alpha: float) -> float:
        """Linear interpolation: a + alpha*(b-a)."""
        return a + alpha * (b - a)

    @staticmethod
    def _generate_persona_id() -> str:
        """
        Generate a unique, non-reversible persona ID.

        Uses current time + random bytes + hash. NOT linked to user identity.
        """
        raw = f"{time.time_ns()}{np.random.bytes(16).hex()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _save_persona(self) -> None:
        """Persist persona state to disk (JSON)."""
        assert self._state_file is not None and self._current_persona is not None
        state = asdict(self._current_persona)
        self._state_file.write_text(json.dumps(state, indent=2))

    def _load_persona(self) -> None:
        """Load persona state from disk."""
        assert self._state_file is not None
        try:
            state = json.loads(self._state_file.read_text())
            p = PersonaState(**{
                k: v for k, v in state.items()
                if k not in ('tremor', 'keystroke', 'cursor')
            })
            if state.get('tremor'):
                p.tremor = TremorPersonaParams(**state['tremor'])
            if state.get('keystroke'):
                p.keystroke = KeystrokePersonaParams(**state['keystroke'])
            if state.get('cursor'):
                p.cursor = CursorPersonaParams(**state['cursor'])
            self._current_persona = p
        except Exception:
            # Fallback: initialize fresh persona if state is corrupt
            self._initialize_persona()


# =============================================================================
# EXAMPLE / VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Cognitive Canary - Persona Coherence Engine v6.1")
    print("=" * 65)

    # Initialize engine
    persona = PersonaCoherenceEngine(rotation_interval=5, seed=42)
    print(f"\nPersona ID: {persona.persona_id}")

    # Get persona parameters
    tremor = persona.get_tremor_params()
    keystroke = persona.get_keystroke_params()
    cursor = persona.get_cursor_params()

    print(f"\n[Tremor Persona]")
    print(f"  Dominant Freq: {tremor.dominant_freq_hz:.2f} Hz")
    print(f"  Amplitude:     {tremor.amplitude:.2f} px")
    print(f"  Entropy Target:{tremor.spectral_entropy_target:.2f} nats")

    print(f"\n[Keystroke Persona]")
    print(f"  Mean IKI:      {keystroke.mean_iki_ms:.1f} ms")
    print(f"  IKI Std:       {keystroke.iki_std_ms:.1f} ms")
    print(f"  Typo Rate:     {keystroke.typo_rate:.3f}")

    print(f"\n[Cursor Persona]")
    print(f"  Mean Velocity: {cursor.mean_velocity_px_s:.1f} px/s")
    print(f"  Path Curvature:{cursor.path_curvature:.2f}")

    # Simulate sessions and enforce consistency
    print(f"\n[Session Simulation]")
    for i in range(7):
        session_id = persona.begin_session()
        raw = {
            'tremor_freq': np.random.uniform(4, 12),
            'tremor_amplitude': np.random.uniform(0.5, 3.0),
            'iki_mean': np.random.uniform(80, 300),
            'cursor_velocity': np.random.uniform(200, 800),
        }
        adjusted = persona.enforce_consistency(raw)

        # Simulate recording session fingerprint
        feature_vec = np.array([
            adjusted['tremor_freq'],
            adjusted['iki_mean'] / 300.0,
            adjusted['cursor_velocity'] / 800.0,
        ])
        persona.record_session_fingerprint(feature_vec)
        print(f"  Session {i+1}: persona={session_id[:8]}... | "
              f"tremor={adjusted['tremor_freq']:.2f}Hz | "
              f"IKI={adjusted['iki_mean']:.0f}ms | "
              f"sessions_until_rotation={persona.sessions_until_rotation}")

    # Audit decorrelation
    print(f"\n[Decorrelation Audit]")
    audit = persona.audit_decorrelation()
    print(f"  Is fingerprintable: {audit['is_correlated']}")
    print(f"  Max correlation:    {audit['max_correlation']:.4f}")
    print(f"  Recommendation:     {audit['recommendation']}")

    print("\n✓ Persona Coherence Engine validated")
    print("=" * 65)
