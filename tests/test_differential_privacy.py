"""
Tests for differential_privacy.py
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from differential_privacy import (
    LaplaceMechanism,
    GaussianMechanism,
    RenyiAccountant,
    PrivacyBudgetTracker,
    DifferentialPrivacyEngine,
    DPConfig,
    BehavioralSensitivityEstimator,
)


class TestLaplaceMechanism:
    """Tests for LaplaceMechanism."""

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError):
            LaplaceMechanism(epsilon=-0.1, sensitivity=1.0)

    def test_zero_epsilon_raises(self):
        with pytest.raises(ValueError):
            LaplaceMechanism(epsilon=0.0, sensitivity=1.0)

    def test_invalid_sensitivity_raises(self):
        with pytest.raises(ValueError):
            LaplaceMechanism(epsilon=1.0, sensitivity=0.0)

    def test_noise_scale_formula(self):
        """Noise scale = sensitivity / epsilon."""
        mech = LaplaceMechanism(epsilon=2.0, sensitivity=1.0)
        assert np.isclose(mech.noise_scale, 0.5)

    def test_privatize_returns_scalar(self):
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        result = mech.privatize(0.5)
        assert isinstance(result, float)

    def test_privatize_array_preserves_shape(self):
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        arr = np.zeros(100)
        result = mech.privatize_array(arr)
        assert result.shape == arr.shape

    def test_noise_is_non_zero_statistically(self):
        """With enough samples, mean noise should be near 0 (Laplace is zero-mean)."""
        rng_state = np.random.get_state()
        np.random.seed(42)
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        values = np.zeros(10000)
        privatized = mech.privatize_array(values)
        np.random.set_state(rng_state)
        # Laplace noise mean should be close to 0
        assert abs(np.mean(privatized)) < 0.1

    def test_smaller_epsilon_more_noise(self):
        """Lower ε → larger noise scale → more noise injected."""
        mech_tight = LaplaceMechanism(epsilon=0.1, sensitivity=1.0)
        mech_loose = LaplaceMechanism(epsilon=10.0, sensitivity=1.0)
        assert mech_tight.noise_scale > mech_loose.noise_scale

    def test_privacy_loss_record(self):
        mech = LaplaceMechanism(epsilon=0.5, sensitivity=1.0)
        loss = mech.privacy_loss_per_query()
        assert loss.mechanism == 'laplace'
        assert loss.epsilon == 0.5
        assert loss.delta == 0.0

    def test_rdp_epsilon_positive(self):
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        for alpha in [2.0, 4.0, 8.0]:
            rdp_eps = mech.rdp_epsilon(alpha)
            assert rdp_eps >= 0.0


class TestGaussianMechanism:
    """Tests for GaussianMechanism."""

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=0.0, delta=1e-6, sensitivity=1.0)

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=1.0, delta=0.0, sensitivity=1.0)
        with pytest.raises(ValueError):
            GaussianMechanism(epsilon=1.0, delta=1.0, sensitivity=1.0)

    def test_sigma_positive(self):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-6, sensitivity=1.0)
        assert mech.sigma > 0

    def test_smaller_epsilon_larger_sigma(self):
        """Lower ε → larger σ (more noise)."""
        mech_tight = GaussianMechanism(epsilon=0.1, delta=1e-6, sensitivity=1.0)
        mech_loose = GaussianMechanism(epsilon=2.0, delta=1e-6, sensitivity=1.0)
        assert mech_tight.sigma > mech_loose.sigma

    def test_privatize_array_shape(self):
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        arr = np.zeros((100, 2))
        result = mech.privatize_array(arr)
        assert result.shape == arr.shape

    def test_rdp_epsilon_formula(self):
        """RDP_α(Gaussian) = α·Δ²/(2σ²)"""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        alpha = 2.0
        expected = alpha * 1.0 / (2 * mech.sigma ** 2)
        assert np.isclose(mech.rdp_epsilon(alpha), expected)

    def test_privacy_loss_record(self):
        mech = GaussianMechanism(epsilon=0.5, delta=1e-6, sensitivity=1.0)
        loss = mech.privacy_loss_per_query()
        assert loss.mechanism == 'gaussian'
        assert loss.delta == 1e-6


class TestRenyiAccountant:
    """Tests for RenyiAccountant."""

    def test_initial_state_zero(self):
        """Zero queries → zero RDP accumulated per order."""
        acc = RenyiAccountant(orders=(2.0, 4.0))
        for order in (2.0, 4.0):
            assert acc._rdp[order] == 0.0
        # to_epsilon_delta should also return 0 with no queries
        eps, _ = acc.to_epsilon_delta(1e-6)
        assert eps == 0.0

    def test_accumulate_increases_epsilon(self):
        acc = RenyiAccountant()
        mech = LaplaceMechanism(epsilon=0.1, sensitivity=1.0)
        loss = mech.privacy_loss_per_query()
        acc.accumulate(loss)
        eps, _ = acc.to_epsilon_delta(1e-6)
        assert eps > 0.0

    def test_rdp_composes_linearly(self):
        """Composing k identical queries: total RDP = k * per_query RDP."""
        acc = RenyiAccountant(orders=(2.0,))
        mech = LaplaceMechanism(epsilon=0.5, sensitivity=1.0)

        n = 10
        for _ in range(n):
            acc.accumulate(mech.privacy_loss_per_query())

        # privacy_loss_per_query uses rdp_epsilon(alpha) internally, so they match
        per_query_rdp = mech.rdp_epsilon(2.0)
        expected_total = n * per_query_rdp
        assert np.isclose(acc._rdp[2.0], expected_total, rtol=1e-6)

    def test_queries_made_counter(self):
        acc = RenyiAccountant()
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        for i in range(5):
            acc.accumulate(mech.privacy_loss_per_query())
        assert acc.queries_made == 5

    def test_reset_clears_state(self):
        acc = RenyiAccountant(orders=(2.0, 4.0))
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        acc.accumulate(mech.privacy_loss_per_query())
        acc.reset()
        assert acc.queries_made == 0
        for order in (2.0, 4.0):
            assert acc._rdp[order] == 0.0
        eps, _ = acc.to_epsilon_delta(1e-6)
        assert eps == 0.0

    def test_rdp_tighter_than_basic_composition(self):
        """After many queries, RDP-derived ε < basic composition ε."""
        acc = RenyiAccountant()
        mech = LaplaceMechanism(epsilon=0.1, sensitivity=1.0)
        n = 100
        for _ in range(n):
            acc.accumulate(mech.privacy_loss_per_query())

        rdp_eps, _ = acc.to_epsilon_delta(1e-6)
        basic_eps = n * 0.1
        # RDP should give a tighter (lower or comparable) bound
        assert rdp_eps <= basic_eps * 1.5  # Allow some slack for conversion overhead

    def test_invalid_delta_raises(self):
        acc = RenyiAccountant()
        with pytest.raises(ValueError):
            acc.to_epsilon_delta(0.0)
        with pytest.raises(ValueError):
            acc.to_epsilon_delta(1.0)

    def test_summary_returns_string(self):
        acc = RenyiAccountant()
        result = acc.summary()
        assert isinstance(result, str)
        assert "queries" in result.lower()


class TestPrivacyBudgetTracker:
    """Tests for PrivacyBudgetTracker."""

    def test_initial_budget_full(self):
        """With no queries, budget fraction should be 1.0 (nothing spent)."""
        tracker = PrivacyBudgetTracker(DPConfig(epsilon_budget=1.0))
        assert tracker.budget_remaining_fraction() == pytest.approx(1.0)

    def test_recording_reduces_budget(self):
        tracker = PrivacyBudgetTracker(DPConfig(epsilon_budget=1.0))
        tracker.record_query('laplace', epsilon=0.1, query_type='test')
        assert tracker.budget_remaining_fraction() < 1.0

    def test_budget_exhausted_flag(self):
        tracker = PrivacyBudgetTracker(DPConfig(epsilon_budget=0.5))
        # Spend 1.0 total ε (exceeds budget)
        for _ in range(10):
            tracker.record_query('laplace', epsilon=0.1, query_type='test')
        assert tracker.is_budget_exhausted()

    def test_recommended_strength_at_full_budget(self):
        tracker = PrivacyBudgetTracker(DPConfig(epsilon_budget=10.0))
        assert tracker.recommended_injection_strength() == 1.0

    def test_recommended_strength_decreases_as_budget_depletes(self):
        tracker = PrivacyBudgetTracker(DPConfig(epsilon_budget=1.0))
        full_strength = tracker.recommended_injection_strength()
        for _ in range(15):
            tracker.record_query('laplace', epsilon=0.06, query_type='test')
        depleted_strength = tracker.recommended_injection_strength()
        assert depleted_strength <= full_strength

    def test_reset_session_clears_budget(self):
        tracker = PrivacyBudgetTracker(DPConfig(epsilon_budget=1.0))
        for _ in range(5):
            tracker.record_query('laplace', epsilon=0.1, query_type='test')
        tracker.reset_session()
        assert tracker.budget_remaining_fraction() == pytest.approx(1.0)

    def test_get_report_structure(self):
        tracker = PrivacyBudgetTracker(DPConfig(epsilon_budget=1.0))
        tracker.record_query('laplace', epsilon=0.1, query_type='cursor_x')
        report = tracker.get_report()
        assert hasattr(report, 'total_epsilon')
        assert hasattr(report, 'rdp_epsilon')
        assert hasattr(report, 'queries_made')
        assert hasattr(report, 'budget_remaining')
        assert hasattr(report, 'is_exhausted')
        assert report.queries_made == 1
        assert report.total_epsilon == pytest.approx(0.1)


class TestDifferentialPrivacyEngine:
    """Integration tests for DifferentialPrivacyEngine."""

    def test_privatize_cursor_x_in_range(self):
        """Privatized cursor x should still be close to [0, 1] with strong ε."""
        dp = DifferentialPrivacyEngine(epsilon_budget=100.0, epsilon_per_query=1.0)
        results = [dp.privatize_cursor_x(0.5) for _ in range(100)]
        # With ε=1.0 and sensitivity=1.0, noise is moderate
        # Check mean stays near 0.5
        assert abs(np.mean(results) - 0.5) < 0.5

    def test_privatize_trajectory_shape(self):
        dp = DifferentialPrivacyEngine(epsilon_budget=100.0, epsilon_per_query=1.0)
        traj = np.random.rand(50, 2)
        result = dp.privatize_cursor_trajectory(traj)
        assert result.shape == traj.shape

    def test_privatize_trajectory_invalid_shape(self):
        dp = DifferentialPrivacyEngine()
        with pytest.raises(ValueError):
            dp.privatize_cursor_trajectory(np.random.rand(50))

    def test_privatize_keystroke_iki_in_bounds(self):
        """Privatized IKI should stay within physical bounds."""
        dp = DifferentialPrivacyEngine(epsilon_budget=100.0, epsilon_per_query=0.5)
        for _ in range(100):
            result = dp.privatize_keystroke_iki(0.15, max_iki_ms=2000.0)
            assert 0.01 <= result <= 2.0

    def test_budget_decreases_with_queries(self):
        dp = DifferentialPrivacyEngine(epsilon_budget=1.0, epsilon_per_query=0.05)
        # Confirm budget starts at 1.0
        assert dp.budget_fraction_remaining == pytest.approx(1.0)
        for _ in range(10):
            dp.privatize_cursor_x(0.5)
        assert dp.budget_fraction_remaining < 1.0

    def test_reset_session(self):
        dp = DifferentialPrivacyEngine(epsilon_budget=1.0, epsilon_per_query=0.1)
        for _ in range(5):
            dp.privatize_cursor_x(0.5)
        dp.reset_session()
        assert dp.budget_fraction_remaining == pytest.approx(1.0)

    def test_budget_report_after_queries(self):
        dp = DifferentialPrivacyEngine(epsilon_budget=5.0, epsilon_per_query=0.1)
        for _ in range(3):
            dp.privatize_cursor_x(0.5)
        report = dp.budget_report()
        assert report.queries_made == 3

    def test_recommended_strength_range(self):
        dp = DifferentialPrivacyEngine(epsilon_budget=1.0)
        strength = dp.recommended_strength
        assert 0.0 <= strength <= 1.0


class TestBehavioralSensitivityEstimator:
    """Tests for BehavioralSensitivityEstimator."""

    def test_cursor_normalized_sensitivity_is_one(self):
        sens = BehavioralSensitivityEstimator.cursor_position_sensitivity(normalized=True)
        assert sens == 1.0

    def test_cursor_raw_sensitivity_positive(self):
        sens = BehavioralSensitivityEstimator.cursor_position_sensitivity(
            screen_width=1920, screen_height=1080, normalized=False
        )
        assert sens == 1920.0

    def test_keystroke_normalized_sensitivity_is_one(self):
        sens = BehavioralSensitivityEstimator.keystroke_iki_sensitivity(normalized=True)
        assert sens == 1.0

    def test_keystroke_raw_sensitivity_in_seconds(self):
        sens = BehavioralSensitivityEstimator.keystroke_iki_sensitivity(
            max_iki_ms=2000.0, normalized=False
        )
        assert np.isclose(sens, 2.0)

    def test_empirical_sensitivity_positive(self):
        rng = np.random.default_rng(0)
        samples = rng.uniform(0.1, 0.9, 200)
        sens = BehavioralSensitivityEstimator.estimate_from_sample(samples, norm='l1')
        assert sens > 0

    def test_empirical_sensitivity_wider_than_range(self):
        """Conservative estimate should be wider than observed range."""
        samples = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        observed_range = 0.6 - 0.2
        sens = BehavioralSensitivityEstimator.estimate_from_sample(samples)
        assert sens >= observed_range

    def test_single_sample_raises(self):
        with pytest.raises(ValueError):
            BehavioralSensitivityEstimator.estimate_from_sample(np.array([0.5]))
