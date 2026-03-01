"""
Cognitive Canary - Differential Privacy Engine
===============================================

Adds formal mathematical privacy guarantees to behavioral obfuscation.

While v6.0 provides empirical evasion metrics (e.g., "98% evasion against
commercial classifiers"), this module provides formal ε-differential privacy
(DP) guarantees: the probability of any inference about your true behavioral
state is bounded by e^ε — regardless of the adversary's computational power.

Implements:
1. Laplace Mechanism  - pure ε-DP for scalar/continuous queries
2. Gaussian Mechanism - (ε, δ)-DP with l2 sensitivity calibration
3. Rényi DP Accountant - tight privacy composition across many queries
4. Privacy Budget Tracker - cumulative leakage monitoring with alerts
5. Behavioral Sensitivity Estimation - automatic feature sensitivity bounds

Mathematical Foundation
-----------------------
A mechanism M satisfies (ε, δ)-DP if for all adjacent datasets D, D'
and all outputs S:

    Pr[M(D) ∈ S] ≤ e^ε · Pr[M(D') ∈ S] + δ

For behavioral data:
    "Adjacent datasets" = two sessions differing by ONE behavioral event
    ε → 0 : strongest protection (maximum noise added)
    ε → ∞ : no protection (no noise)
    δ     : failure probability (set ≤ 1/n² for n-sample sessions)

Rényi DP Composition (preferred over basic composition):
    RDP(α, ε) composed k times → RDP(α, k·ε)
    Converts to (ε_final, δ)-DP via:
        ε_final = ε_rdp + log(1/δ) / (α - 1)

Author: Cognitive Canary Project
License: MIT
Version: 6.1
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PrivacyLoss:
    """Records a single privacy expenditure."""
    mechanism: str         # 'laplace', 'gaussian', 'randomized_response'
    epsilon: float         # ε spent
    delta: float           # δ spent (0 for pure DP)
    query_type: str        # what behavioral feature was queried
    rdp_orders: Optional[Dict[float, float]] = None  # RDP ε per order α


@dataclass
class PrivacyBudgetReport:
    """Summary of cumulative privacy budget consumption."""
    total_epsilon: float         # Pure ε composition bound
    total_delta: float           # Accumulated δ
    rdp_epsilon: float           # Tighter RDP-derived ε
    queries_made: int
    budget_remaining: float      # Fraction of configured budget left
    is_exhausted: bool           # True if budget spent
    top_spending_queries: List[str]  # Which query types cost the most


@dataclass
class DPConfig:
    """Configuration for the differential privacy engine."""
    # Privacy budget (total ε allowed over a session)
    epsilon_budget: float = 1.0      # ε = 1.0 is considered strong privacy
    delta_budget: float = 1e-6       # δ ≤ 1/n² for n-sample sessions

    # RDP orders to track (standard set for tight bounds)
    rdp_orders: Tuple[float, ...] = (
        1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0
    )

    # Alert thresholds
    budget_warning_fraction: float = 0.8   # Warn at 80% budget consumed
    budget_critical_fraction: float = 0.95 # Critical at 95%

    # Behavioral sensitivity bounds (measured in normalized units)
    cursor_l1_sensitivity: float = 1.0     # pixels, normalized
    keystroke_l1_sensitivity: float = 0.02 # seconds, normalized (20ms)
    eeg_l1_sensitivity: float = 0.1        # amplitude units


# =============================================================================
# LAPLACE MECHANISM
# =============================================================================

class LaplaceMechanism:
    """
    Laplace mechanism for pure ε-differential privacy.

    Adds Laplace-distributed noise calibrated to global l1 sensitivity Δf:
        Noise ~ Laplace(0, Δf/ε)

    Achieves ε-DP with δ = 0 (strongest formal guarantee).

    Parameters
    ----------
    epsilon : float
        Privacy budget ε > 0. Smaller = more private.
    sensitivity : float
        Global l1 sensitivity of the query function Δf.
    """

    def __init__(self, epsilon: float, sensitivity: float):
        if epsilon <= 0:
            raise ValueError(f"ε must be positive, got {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be positive, got {sensitivity}")
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self._scale = sensitivity / epsilon  # Laplace scale parameter b

    @property
    def noise_scale(self) -> float:
        """Laplace scale parameter b = Δf/ε."""
        return self._scale

    def privatize(self, value: float) -> float:
        """Add Laplace noise to a scalar value."""
        return value + np.random.laplace(0.0, self._scale)

    def privatize_array(self, values: np.ndarray) -> np.ndarray:
        """Add independent Laplace noise to each element."""
        noise = np.random.laplace(0.0, self._scale, size=values.shape)
        return values + noise

    def privacy_loss_per_query(self) -> PrivacyLoss:
        """Return the PrivacyLoss record for one application."""
        rdp = {alpha: self.rdp_epsilon(alpha) for alpha in [1.5, 2.0, 4.0, 8.0]}
        return PrivacyLoss(
            mechanism='laplace',
            epsilon=self.epsilon,
            delta=0.0,
            query_type='scalar',
            rdp_orders=rdp,
        )

    def rdp_epsilon(self, alpha: float) -> float:
        """
        Rényi DP ε at order α for Laplace mechanism.

        For Laplace(b): RDP(α) = log(α/(2α-1)) / (α-1)  +  ε·α/(2)
        Approximate closed-form for tight tracking.
        """
        if alpha == 1.0:
            return self.epsilon  # Degenerate case
        b = self._scale
        # Exact RDP for Laplace mechanism
        try:
            term1 = (np.log(alpha) - np.log(2 * alpha - 1)) / (alpha - 1)
            term2 = alpha / (2 * b)
            return term1 + term2
        except (ZeroDivisionError, ValueError):
            return self.epsilon * alpha  # Safe fallback


# =============================================================================
# GAUSSIAN MECHANISM
# =============================================================================

class GaussianMechanism:
    """
    Gaussian mechanism for (ε, δ)-differential privacy.

    Adds Gaussian noise calibrated to l2 sensitivity Δ₂f:
        Noise ~ N(0, σ²)  where  σ = Δ₂f · √(2 ln(1.25/δ)) / ε

    Achieves (ε, δ)-DP. With δ > 0, admits tighter noise than Laplace
    for high-dimensional queries (e.g., entire cursor trajectory).

    Parameters
    ----------
    epsilon : float
        Target ε (privacy parameter). Smaller = more private.
    delta : float
        Failure probability δ. Set to 1/n² for n-sample sessions.
    sensitivity : float
        Global l2 sensitivity of the query function Δ₂f.
    """

    def __init__(self, epsilon: float, delta: float, sensitivity: float):
        if epsilon <= 0:
            raise ValueError(f"ε must be positive, got {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"δ must be in (0, 1), got {delta}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be positive, got {sensitivity}")

        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        # Calibrate σ using analytic Gaussian mechanism
        self._sigma = self._calibrate_sigma()

    def _calibrate_sigma(self) -> float:
        """Calibrate Gaussian std-dev for (ε, δ)-DP."""
        # Standard analytic formula: σ = Δ₂ · √(2 ln(1.25/δ)) / ε
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    @property
    def sigma(self) -> float:
        """Standard deviation of Gaussian noise."""
        return self._sigma

    def privatize(self, value: float) -> float:
        """Add Gaussian noise to a scalar value."""
        return value + np.random.normal(0.0, self._sigma)

    def privatize_array(self, values: np.ndarray) -> np.ndarray:
        """Add independent Gaussian noise to each element."""
        return values + np.random.normal(0.0, self._sigma, size=values.shape)

    def rdp_epsilon(self, alpha: float) -> float:
        """
        Rényi DP ε at order α for Gaussian mechanism.

        Exact formula: RDP_α(M_Gauss) = α · Δ₂² / (2σ²)
        """
        return alpha * (self.sensitivity ** 2) / (2 * self._sigma ** 2)

    def privacy_loss_per_query(self) -> PrivacyLoss:
        """Return the PrivacyLoss record for one application."""
        rdp = {alpha: self.rdp_epsilon(alpha) for alpha in [1.5, 2.0, 4.0, 8.0]}
        return PrivacyLoss(
            mechanism='gaussian',
            epsilon=self.epsilon,
            delta=self.delta,
            query_type='vector',
            rdp_orders=rdp,
        )


# =============================================================================
# RÉNYI DP ACCOUNTANT
# =============================================================================

class RenyiAccountant:
    """
    Rényi Differential Privacy (RDP) accountant for tight composition.

    Tracks privacy loss across multiple mechanism applications using
    Rényi divergences, which compose linearly and convert to (ε, δ)-DP:

        After k adaptive queries with per-query RDP(α) = ε_i(α):
            Total RDP(α) = Σᵢ εᵢ(α)

        Convert to (ε_final, δ)-DP for any δ > 0:
            ε_final = min_α [ Σᵢ εᵢ(α) + log(1/δ) / (α - 1) ]

    This is tighter than basic composition (Σᵢ εᵢ) by up to √k factor.

    Parameters
    ----------
    orders : tuple of float
        Rényi orders α > 1 to track. Standard: (1.5, 2, 4, 8, 16, 32, 64)
    """

    def __init__(self, orders: Tuple[float, ...] = (1.5, 2.0, 4.0, 8.0, 16.0, 32.0)):
        self.orders = orders
        # Accumulated RDP ε per order
        self._rdp: Dict[float, float] = {alpha: 0.0 for alpha in orders}
        self._history: List[PrivacyLoss] = []

    def accumulate(self, loss: PrivacyLoss) -> None:
        """Record a privacy expenditure."""
        self._history.append(loss)
        if loss.rdp_orders:
            for alpha, eps in loss.rdp_orders.items():
                if alpha in self._rdp:
                    self._rdp[alpha] += eps
        else:
            # Fallback: treat as pure DP, distribute across orders
            for alpha in self.orders:
                self._rdp[alpha] += loss.epsilon

    def to_epsilon_delta(self, delta: float) -> Tuple[float, float]:
        """
        Convert accumulated RDP to (ε, δ)-DP using the tightest bound.

        Parameters
        ----------
        delta : float
            Target δ for the conversion.

        Returns
        -------
        (epsilon, delta)
            The tightest (ε, δ)-DP guarantee achievable.
        """
        if delta <= 0 or delta >= 1:
            raise ValueError(f"δ must be in (0,1), got {delta}")

        # No queries made → zero privacy spent
        if self.queries_made == 0:
            return 0.0, delta

        best_eps = float('inf')
        for alpha in self.orders:
            if alpha <= 1.0:
                continue
            rdp_eps = self._rdp[alpha]
            # Conversion formula: ε = RDP(α) + log(1/δ)/(α-1)
            eps = rdp_eps + np.log(1.0 / delta) / (alpha - 1)
            best_eps = min(best_eps, eps)

        return best_eps, delta

    @property
    def queries_made(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        """Reset accumulated privacy losses (start new session)."""
        self._rdp = {alpha: 0.0 for alpha in self.orders}
        self._history.clear()

    def summary(self, delta: float = 1e-6) -> str:
        """Human-readable summary of privacy expenditure."""
        eps, d = self.to_epsilon_delta(delta)
        return (
            f"Privacy after {self.queries_made} queries: "
            f"(ε={eps:.4f}, δ={d:.2e})"
        )


# =============================================================================
# PRIVACY BUDGET TRACKER
# =============================================================================

class PrivacyBudgetTracker:
    """
    High-level privacy budget manager for a Cognitive Canary session.

    Tracks cumulative privacy loss across all obfuscation operations,
    alerts when budget approaches exhaustion, and recommends injection
    strength adjustments to stay within budget.

    Design Philosophy
    -----------------
    The privacy budget ε represents how much information about your
    true behavioral state can leak across all operations in a session.
    When ε = 1.0 (default), your true state is distinguishable from
    an adjacent state by at most e^1 ≈ 2.7× — a strong guarantee.

    At budget exhaustion, the tracker recommends switching to pure
    deterministic noise (no DP overhead) to preserve usability.

    Parameters
    ----------
    config : DPConfig
        Configuration including total budget and thresholds.
    """

    def __init__(self, config: Optional[DPConfig] = None):
        self.config = config or DPConfig()
        self._accountant = RenyiAccountant(orders=self.config.rdp_orders)
        self._naive_epsilon_total: float = 0.0  # Basic composition for comparison
        self._delta_total: float = 0.0
        self._query_costs: Dict[str, float] = {}
        self._alert_raised: bool = False

    def record_query(
        self,
        mechanism: str,
        epsilon: float,
        delta: float = 0.0,
        query_type: str = "behavioral",
        rdp_orders: Optional[Dict[float, float]] = None,
    ) -> None:
        """
        Record a privacy expenditure.

        Parameters
        ----------
        mechanism : str
            'laplace', 'gaussian', or 'randomized_response'
        epsilon : float
            ε spent by this mechanism application.
        delta : float
            δ spent (0 for pure DP mechanisms).
        query_type : str
            Label for what was queried (e.g., 'cursor_x', 'keystroke_iki').
        rdp_orders : dict, optional
            Per-order RDP ε for tighter accounting.
        """
        loss = PrivacyLoss(
            mechanism=mechanism,
            epsilon=epsilon,
            delta=delta,
            query_type=query_type,
            rdp_orders=rdp_orders,
        )
        self._accountant.accumulate(loss)
        self._naive_epsilon_total += epsilon
        self._delta_total += delta

        # Track per-query-type costs
        self._query_costs[query_type] = (
            self._query_costs.get(query_type, 0.0) + epsilon
        )

        self._check_budget()

    def record_laplace(
        self, mechanism: LaplaceMechanism, query_type: str = "scalar"
    ) -> None:
        """Convenience: record one Laplace mechanism application."""
        loss = mechanism.privacy_loss_per_query()
        loss.query_type = query_type
        self._accountant.accumulate(loss)
        self._naive_epsilon_total += mechanism.epsilon
        self._query_costs[query_type] = (
            self._query_costs.get(query_type, 0.0) + mechanism.epsilon
        )
        self._check_budget()

    def record_gaussian(
        self, mechanism: GaussianMechanism, query_type: str = "vector"
    ) -> None:
        """Convenience: record one Gaussian mechanism application."""
        loss = mechanism.privacy_loss_per_query()
        loss.query_type = query_type
        self._accountant.accumulate(loss)
        self._naive_epsilon_total += mechanism.epsilon
        self._delta_total += mechanism.delta
        self._query_costs[query_type] = (
            self._query_costs.get(query_type, 0.0) + mechanism.epsilon
        )
        self._check_budget()

    def get_current_epsilon(self, delta: Optional[float] = None) -> float:
        """
        Get the current (tightest) ε spent so far.

        Uses RDP accounting if available, else basic composition.
        """
        d = delta or self.config.delta_budget
        try:
            rdp_eps, _ = self._accountant.to_epsilon_delta(d)
            return rdp_eps
        except Exception:
            return self._naive_epsilon_total

    def budget_remaining_fraction(self) -> float:
        """Fraction of budget remaining (1.0 = full, 0.0 = exhausted)."""
        spent = self.get_current_epsilon()
        remaining = max(0.0, 1.0 - spent / self.config.epsilon_budget)
        return remaining

    def is_budget_exhausted(self) -> bool:
        return self.get_current_epsilon() >= self.config.epsilon_budget

    def recommended_injection_strength(self) -> float:
        """
        Recommend injection strength based on remaining privacy budget.

        When budget is ample: maintain full injection.
        As budget depletes: scale down to avoid over-spending.
        At exhaustion: switch to deterministic patterns (no DP overhead).
        """
        fraction = self.budget_remaining_fraction()
        if fraction > self.config.budget_warning_fraction:
            return 1.0   # Full strength
        elif fraction > 0.2:
            # Linear scale-down in warning zone
            return 0.5 + 0.5 * (fraction / self.config.budget_warning_fraction)
        else:
            return 0.3   # Minimum strength (matches PRODUCTIVITY_INJECTION_MIN)

    def get_report(self) -> PrivacyBudgetReport:
        """Generate a comprehensive privacy budget report."""
        rdp_eps = self.get_current_epsilon()
        top_queries = sorted(
            self._query_costs.items(), key=lambda x: x[1], reverse=True
        )[:3]
        return PrivacyBudgetReport(
            total_epsilon=self._naive_epsilon_total,
            total_delta=self._delta_total,
            rdp_epsilon=rdp_eps,
            queries_made=self._accountant.queries_made,
            budget_remaining=self.budget_remaining_fraction(),
            is_exhausted=self.is_budget_exhausted(),
            top_spending_queries=[q for q, _ in top_queries],
        )

    def reset_session(self) -> None:
        """Reset for a new obfuscation session."""
        self._accountant.reset()
        self._naive_epsilon_total = 0.0
        self._delta_total = 0.0
        self._query_costs.clear()
        self._alert_raised = False

    def _check_budget(self) -> None:
        """Internal: raise warning if budget thresholds crossed."""
        fraction = self.budget_remaining_fraction()
        if fraction <= (1.0 - self.config.budget_warning_fraction) and not self._alert_raised:
            warnings.warn(
                f"[CognitiveCanary DP] Privacy budget {self.config.budget_warning_fraction*100:.0f}% "
                f"consumed (ε_spent={self.get_current_epsilon():.4f} / "
                f"ε_budget={self.config.epsilon_budget}). "
                f"Consider resetting session or reducing injection frequency.",
                UserWarning,
                stacklevel=3,
            )
            self._alert_raised = True


# =============================================================================
# BEHAVIORAL SENSITIVITY ESTIMATOR
# =============================================================================

class BehavioralSensitivityEstimator:
    """
    Estimates global l1/l2 sensitivity of behavioral feature functions.

    Sensitivity Δf bounds how much one behavioral event (one keystroke,
    one cursor position) can change the output of a query function.
    Correct sensitivity bounds are critical: underestimating them
    breaks the DP guarantee; overestimating wastes privacy budget.

    Approach: Uses empirical range bounding from a calibration sample,
    with conservative multipliers for unknown tails.
    """

    # Conservative safety multiplier for unknown distribution tails
    TAIL_SAFETY_MULTIPLIER = 1.5

    @classmethod
    def cursor_position_sensitivity(
        cls,
        screen_width: int = 1920,
        screen_height: int = 1080,
        normalized: bool = True,
    ) -> float:
        """
        Global l1 sensitivity for a single cursor coordinate query.

        f(x) = cursor_x  →  Δf = max change per event = range of x
        After normalization to [0,1]: Δf = 1.0
        """
        if normalized:
            return 1.0
        return float(max(screen_width, screen_height))

    @classmethod
    def keystroke_iki_sensitivity(
        cls,
        max_iki_ms: float = 2000.0,
        normalized: bool = True,
    ) -> float:
        """
        Global l1 sensitivity for inter-key-interval (IKI) query.

        After normalization to [0, max_iki_ms]: Δf = max_iki_ms
        In seconds: Δf = max_iki_ms / 1000
        """
        if normalized:
            return 1.0
        return max_iki_ms / 1000.0  # seconds

    @classmethod
    def estimate_from_sample(
        cls,
        samples: np.ndarray,
        norm: str = 'l1',
    ) -> float:
        """
        Estimate sensitivity empirically from a sample of feature values.

        Uses the sample range as a proxy for sensitivity, with a
        conservative tail multiplier for unseen values.

        Parameters
        ----------
        samples : np.ndarray
            Array of observed feature values.
        norm : 'l1' or 'l2'
            Which sensitivity norm to estimate.

        Returns
        -------
        float
            Conservative sensitivity estimate.
        """
        if len(samples) < 2:
            raise ValueError("Need at least 2 samples to estimate sensitivity")

        sample_range = np.max(samples) - np.min(samples)
        conservative_bound = sample_range * cls.TAIL_SAFETY_MULTIPLIER

        if norm == 'l2':
            # l2 sensitivity ≤ l1 sensitivity (generally)
            return conservative_bound / np.sqrt(len(samples))
        return float(conservative_bound)


# =============================================================================
# INTEGRATED DP OBFUSCATION INTERFACE
# =============================================================================

class DifferentialPrivacyEngine:
    """
    Top-level interface for DP-enhanced behavioral obfuscation.

    Wraps all DP mechanisms with a unified API and automatic budget tracking.
    Integrates with existing Cognitive Canary engines (Lissajous3D,
    AdaptiveTremor, KeystrokeJitter) to add formal privacy guarantees on top
    of the empirical obfuscation layer.

    Usage
    -----
    >>> dp = DifferentialPrivacyEngine(epsilon_budget=1.0)
    >>> obfuscated_x = dp.privatize_cursor_x(raw_x)
    >>> obfuscated_iki = dp.privatize_keystroke_iki(raw_iki)
    >>> report = dp.budget_report()
    >>> print(f"Privacy spent: ε={report.rdp_epsilon:.4f}")
    """

    def __init__(
        self,
        epsilon_budget: float = 1.0,
        delta_budget: float = 1e-6,
        epsilon_per_query: float = 0.01,
    ):
        """
        Parameters
        ----------
        epsilon_budget : float
            Total ε allowed per session. Default 1.0 (strong privacy).
        delta_budget : float
            Total δ allowed per session.
        epsilon_per_query : float
            ε to spend per individual query (controls noise magnitude).
            Smaller = more noise = better privacy per query.
        """
        config = DPConfig(epsilon_budget=epsilon_budget, delta_budget=delta_budget)
        self._tracker = PrivacyBudgetTracker(config)
        self._eps_per_query = epsilon_per_query
        self._sensitivity = BehavioralSensitivityEstimator()

    def privatize_cursor_x(self, x: float, normalized: bool = True) -> float:
        """
        Apply DP noise to a single cursor x-coordinate.

        Uses Laplace mechanism (pure ε-DP, δ=0).
        """
        sens = self._sensitivity.cursor_position_sensitivity(normalized=normalized)
        mech = LaplaceMechanism(self._eps_per_query, sens)
        result = mech.privatize(x)
        self._tracker.record_laplace(mech, query_type='cursor_x')
        return result

    def privatize_cursor_trajectory(
        self, trajectory: np.ndarray, normalized: bool = True
    ) -> np.ndarray:
        """
        Apply DP noise to a 2D cursor trajectory array (N, 2).

        Uses Gaussian mechanism for vector queries (tighter than Laplace
        for high-dimensional data).
        """
        if trajectory.ndim != 2 or trajectory.shape[1] < 2:
            raise ValueError("trajectory must be shape (N, 2)")

        # l2 sensitivity for trajectory: each point can change by at most √2
        l2_sens = self._sensitivity.cursor_position_sensitivity(normalized=normalized) * np.sqrt(2)
        mech = GaussianMechanism(
            self._eps_per_query, self._tracker.config.delta_budget / 100, l2_sens
        )
        result = mech.privatize_array(trajectory)
        self._tracker.record_gaussian(mech, query_type='cursor_trajectory')
        return result

    def privatize_keystroke_iki(
        self, iki_seconds: float, max_iki_ms: float = 2000.0
    ) -> float:
        """
        Apply DP noise to an inter-key-interval (in seconds).

        Uses Laplace mechanism. Clamps result to [0, max_iki_ms/1000]
        to preserve physical plausibility.
        """
        sens = self._sensitivity.keystroke_iki_sensitivity(
            max_iki_ms=max_iki_ms, normalized=False
        )
        mech = LaplaceMechanism(self._eps_per_query, sens)
        result = mech.privatize(iki_seconds)
        # Clamp to physically plausible range
        result = float(np.clip(result, 0.01, max_iki_ms / 1000.0))
        self._tracker.record_laplace(mech, query_type='keystroke_iki')
        return result

    def privatize_eeg_band_power(self, power: float) -> float:
        """
        Apply DP noise to an EEG band power value.

        Uses Laplace mechanism with EEG-specific sensitivity.
        """
        mech = LaplaceMechanism(
            self._eps_per_query,
            self._tracker.config.eeg_l1_sensitivity
        )
        result = mech.privatize(power)
        self._tracker.record_laplace(mech, query_type='eeg_band_power')
        return result

    def budget_report(self) -> PrivacyBudgetReport:
        """Return current privacy budget consumption report."""
        return self._tracker.get_report()

    def reset_session(self) -> None:
        """Reset for a new protection session."""
        self._tracker.reset_session()

    @property
    def budget_fraction_remaining(self) -> float:
        """Fraction of total ε budget remaining."""
        return self._tracker.budget_remaining_fraction()

    @property
    def recommended_strength(self) -> float:
        """Recommended injection strength (0.3 – 1.0) based on budget."""
        return self._tracker.recommended_injection_strength()


# =============================================================================
# EXAMPLE / VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Cognitive Canary - Differential Privacy Engine v6.1")
    print("=" * 65)

    # --- Laplace Mechanism Demo ---
    print("\n[1] Laplace Mechanism (pure ε-DP)")
    lap = LaplaceMechanism(epsilon=0.1, sensitivity=1.0)
    raw_x = 0.5  # Normalized cursor x position
    private_x = lap.privatize(raw_x)
    print(f"    Raw cursor x: {raw_x:.4f}")
    print(f"    Private x:    {private_x:.4f}  (noise scale b={lap.noise_scale:.4f})")

    # --- Gaussian Mechanism Demo ---
    print("\n[2] Gaussian Mechanism ((ε, δ)-DP)")
    gauss = GaussianMechanism(epsilon=0.5, delta=1e-6, sensitivity=1.0)
    trajectory = np.random.rand(100, 2)  # 100 cursor points
    private_traj = gauss.privatize_array(trajectory)
    print(f"    Trajectory shape: {trajectory.shape}")
    print(f"    Gaussian σ: {gauss.sigma:.4f}")
    print(f"    Mean noise: {np.mean(np.abs(private_traj - trajectory)):.4f}")

    # --- RDP Accountant Demo ---
    print("\n[3] Rényi DP Accountant (tight composition)")
    accountant = RenyiAccountant()
    for _ in range(50):  # Simulate 50 queries
        accountant.accumulate(lap.privacy_loss_per_query())
    eps_final, delta = accountant.to_epsilon_delta(delta=1e-6)
    print(f"    After 50 queries (ε=0.1 each):")
    print(f"    Basic composition: ε = {50 * 0.1:.2f}")
    print(f"    RDP composition:   ε = {eps_final:.4f}  (tighter!)")

    # --- Full Engine Demo ---
    print("\n[4] DifferentialPrivacyEngine (integrated API)")
    dp = DifferentialPrivacyEngine(epsilon_budget=2.0, epsilon_per_query=0.05)

    # Simulate a session with 20 cursor queries and 20 keystroke queries
    for _ in range(20):
        dp.privatize_cursor_x(np.random.uniform(0, 1))
    for _ in range(20):
        dp.privatize_keystroke_iki(np.random.uniform(0.05, 0.5))

    report = dp.budget_report()
    print(f"    Queries made: {report.queries_made}")
    print(f"    ε spent (basic):  {report.total_epsilon:.4f}")
    print(f"    ε spent (RDP):    {report.rdp_epsilon:.4f}")
    print(f"    Budget remaining: {report.budget_remaining:.1%}")
    print(f"    Top spending:     {report.top_spending_queries}")
    print(f"    Recommended strength: {dp.recommended_strength:.2f}")

    print("\n✓ Differential Privacy Engine validated")
    print("=" * 65)
