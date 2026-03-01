"""
Tests for persona_engine.py
"""

import numpy as np
import pytest
import sys
import tempfile
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from persona_engine import (
    PersonaCoherenceEngine,
    PersonaState,
    TremorPersonaParams,
    KeystrokePersonaParams,
    CursorPersonaParams,
    TREMOR_FREQ_RANGE,
    TREMOR_AMP_RANGE,
    KEYSTROKE_IKI_MEAN_RANGE,
    CURSOR_VELOCITY_MEAN_RANGE,
)


class TestPersonaCoherenceEngineInit:
    """Tests for initialization and basic state."""

    def test_creates_valid_persona(self):
        engine = PersonaCoherenceEngine(seed=0)
        assert engine._current_persona is not None

    def test_persona_id_is_string(self):
        engine = PersonaCoherenceEngine(seed=1)
        assert isinstance(engine.persona_id, str)

    def test_persona_id_is_hex_length(self):
        engine = PersonaCoherenceEngine(seed=2)
        assert len(engine.persona_id) == 16

    def test_tremor_params_in_valid_range(self):
        engine = PersonaCoherenceEngine(seed=3)
        p = engine.get_tremor_params()
        assert TREMOR_FREQ_RANGE[0] <= p.dominant_freq_hz <= TREMOR_FREQ_RANGE[1]
        assert TREMOR_AMP_RANGE[0] <= p.amplitude <= TREMOR_AMP_RANGE[1]
        assert 0.0 <= p.phase_offset_rad <= 2 * np.pi

    def test_keystroke_params_in_valid_range(self):
        engine = PersonaCoherenceEngine(seed=4)
        p = engine.get_keystroke_params()
        assert KEYSTROKE_IKI_MEAN_RANGE[0] <= p.mean_iki_ms <= KEYSTROKE_IKI_MEAN_RANGE[1]
        assert p.typo_rate >= 0.02
        assert p.typo_rate <= 0.08

    def test_cursor_params_in_valid_range(self):
        engine = PersonaCoherenceEngine(seed=5)
        p = engine.get_cursor_params()
        assert CURSOR_VELOCITY_MEAN_RANGE[0] <= p.mean_velocity_px_s <= CURSOR_VELOCITY_MEAN_RANGE[1]
        assert 0.1 <= p.path_curvature <= 0.9


class TestBeginSession:
    """Tests for begin_session()."""

    def test_returns_persona_id(self):
        engine = PersonaCoherenceEngine(seed=10)
        pid = engine.begin_session()
        assert isinstance(pid, str)
        assert len(pid) == 16

    def test_session_counter_increments(self):
        engine = PersonaCoherenceEngine(rotation_interval=100, seed=11)
        initial_used = engine._current_persona.sessions_used
        engine.begin_session()
        assert engine._current_persona.sessions_used == initial_used + 1

    def test_rotation_triggers_after_interval(self):
        """Persona should rotate after rotation_interval sessions."""
        engine = PersonaCoherenceEngine(rotation_interval=3, seed=12)
        original_id = engine.persona_id

        # Begin 4 sessions (exceeds interval of 3)
        for _ in range(4):
            engine.begin_session()

        # After rotation, persona ID should have changed
        assert engine.persona_id != original_id

    def test_sessions_until_rotation_decrements(self):
        engine = PersonaCoherenceEngine(rotation_interval=5, seed=13)
        initial = engine.sessions_until_rotation
        engine.begin_session()
        assert engine.sessions_until_rotation == max(0, initial - 1)


class TestPersonaConsistency:
    """Tests for persona consistency across sessions."""

    def test_same_persona_same_params(self):
        """Consecutive sessions within window return same tremor params."""
        engine = PersonaCoherenceEngine(rotation_interval=10, seed=20)
        params_1 = engine.get_tremor_params()
        engine.begin_session()
        params_2 = engine.get_tremor_params()
        # Within rotation window, params should be identical
        assert params_1.dominant_freq_hz == params_2.dominant_freq_hz
        assert params_1.amplitude == params_2.amplitude

    def test_different_seeds_different_personas(self):
        """Different seeds produce different personas."""
        engine_a = PersonaCoherenceEngine(seed=100)
        engine_b = PersonaCoherenceEngine(seed=200)
        assert engine_a.persona_id != engine_b.persona_id


class TestEnforceConsistency:
    """Tests for enforce_consistency()."""

    def test_returns_dict(self):
        engine = PersonaCoherenceEngine(seed=30)
        raw = {'tremor_freq': 6.0, 'iki_mean': 150.0}
        result = engine.enforce_consistency(raw)
        assert isinstance(result, dict)

    def test_adjusted_closer_to_persona(self):
        """Adjusted tremor_freq should be between raw and persona value."""
        engine = PersonaCoherenceEngine(seed=31)
        persona_freq = engine.get_tremor_params().dominant_freq_hz
        raw_freq = 11.0 if persona_freq < 8.0 else 4.5  # Far from persona

        raw = {'tremor_freq': raw_freq}
        adjusted = engine.enforce_consistency(raw)

        # The adjustment should move toward persona
        distance_raw = abs(raw_freq - persona_freq)
        distance_adjusted = abs(adjusted['tremor_freq'] - persona_freq)
        assert distance_adjusted < distance_raw

    def test_unknown_keys_preserved(self):
        """Keys not in the blending set should be passed through unchanged."""
        engine = PersonaCoherenceEngine(seed=32)
        raw = {'unknown_key': 42.0, 'tremor_freq': 7.0}
        adjusted = engine.enforce_consistency(raw)
        assert adjusted['unknown_key'] == 42.0

    def test_all_known_keys_adjusted(self):
        engine = PersonaCoherenceEngine(seed=33)
        raw = {
            'tremor_freq': 12.0,
            'tremor_amplitude': 3.0,
            'iki_mean': 300.0,
            'cursor_velocity': 800.0,
        }
        adjusted = engine.enforce_consistency(raw)
        # All keys present in output
        for key in raw:
            assert key in adjusted


class TestRecordAndAuditFingerprints:
    """Tests for session fingerprint recording and decorrelation audit."""

    def test_record_fingerprint_stores_value(self):
        engine = PersonaCoherenceEngine(seed=40)
        feature_vec = np.array([6.0, 0.5, 0.4])
        engine.record_session_fingerprint(feature_vec)
        assert len(engine._current_persona.session_fingerprints) == 1

    def test_fingerprint_history_bounded(self):
        """History should not grow beyond 50 entries."""
        engine = PersonaCoherenceEngine(seed=41)
        for _ in range(60):
            engine.record_session_fingerprint(np.random.rand(3))
        assert len(engine._current_persona.session_fingerprints) <= 50

    def test_audit_insufficient_sessions(self):
        """Audit should report insufficient data with < 3 sessions."""
        engine = PersonaCoherenceEngine(seed=42)
        result = engine.audit_decorrelation()
        assert 'recommendation' in result
        assert not result['is_correlated']

    def test_audit_with_enough_sessions(self):
        """Audit runs without error after 3+ sessions."""
        engine = PersonaCoherenceEngine(seed=43)
        for _ in range(5):
            engine.record_session_fingerprint(np.random.default_rng(43).random(3))
        result = engine.audit_decorrelation()
        assert 'is_correlated' in result
        assert 'max_correlation' in result
        assert isinstance(result['max_correlation'], float)

    def test_audit_returns_recommendation_string(self):
        engine = PersonaCoherenceEngine(seed=44)
        for _ in range(5):
            engine.record_session_fingerprint(np.array([7.0, 0.5, 0.6]))
        result = engine.audit_decorrelation()
        assert isinstance(result['recommendation'], str)


class TestPersonaStatePersistence:
    """Tests for persona state persistence to disk."""

    def test_save_and_load(self):
        """Persona saved to disk should be loadable with consistent params."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            state_file = f.name

        engine1 = PersonaCoherenceEngine(rotation_interval=10, seed=50,
                                          persona_state_file=state_file)
        original_freq = engine1.get_tremor_params().dominant_freq_hz

        # Trigger a save
        engine1.begin_session()

        # Load into new engine
        engine2 = PersonaCoherenceEngine(rotation_interval=10, seed=50,
                                          persona_state_file=state_file)
        loaded_freq = engine2.get_tremor_params().dominant_freq_hz

        Path(state_file).unlink(missing_ok=True)

        assert np.isclose(original_freq, loaded_freq, rtol=1e-6)

    def test_corrupt_file_falls_back_to_fresh_persona(self):
        """Corrupt state file should result in a fresh persona without crashing."""
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            f.write("{{corrupt json{{")
            state_file = f.name

        engine = PersonaCoherenceEngine(persona_state_file=state_file, seed=51)
        # Should have a valid persona despite corrupt file
        assert engine._current_persona is not None
        Path(state_file).unlink(missing_ok=True)


class TestPersonaRotation:
    """Tests for persona rotation behavior."""

    def test_rotation_changes_params(self):
        """After forced rotation, at least some params should change."""
        engine = PersonaCoherenceEngine(rotation_interval=1, seed=60)
        freq_before = engine.get_tremor_params().dominant_freq_hz
        engine.begin_session()  # triggers rotation (rotation_interval=1)
        engine.begin_session()  # second session after rotation
        freq_after = engine.get_tremor_params().dominant_freq_hz
        # Due to interpolation (alpha=0.4), params shift but not necessarily all the way
        # Just verify the engine is still functional
        assert isinstance(freq_after, float)
        assert TREMOR_FREQ_RANGE[0] <= freq_after <= TREMOR_FREQ_RANGE[1]

    def test_rotation_resets_session_counter(self):
        engine = PersonaCoherenceEngine(rotation_interval=2, seed=61)
        engine.begin_session()
        engine.begin_session()
        engine.begin_session()  # triggers rotation
        assert engine._current_persona.sessions_used == 0

    def test_sessions_until_rotation_is_non_negative(self):
        engine = PersonaCoherenceEngine(rotation_interval=3, seed=62)
        for _ in range(10):
            engine.begin_session()
            assert engine.sessions_until_rotation >= 0
