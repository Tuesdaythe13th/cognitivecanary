"""
Cognitive Canary - Neuro Rights Audit Engine
=============================================

Scans behavioral and neural data collection for compliance with 2026
neurorights legislation, flags violations, and generates legal compliance
reports suitable for regulatory submission.

The 2026 Legislative Landscape
--------------------------------
Unlike GDPR (a single unified standard), neurorights law in 2026 is
catastrophically fragmented. Cognitive Canary users may operate across
jurisdictions with radically different protections:

    ┌─────────────────────┬──────────────────────────────────────────────┐
    │ Jurisdiction        │ Key Requirement                              │
    ├─────────────────────┼──────────────────────────────────────────────┤
    │ Chile               │ Constitutional protection for mental privacy, │
    │                     │ identity, free will. Neural data ≠ commodity. │
    │ Colorado (USA)      │ Neural data = "sensitive personal data".      │
    │                     │ Explicit consent + opt-out required.          │
    │ UNESCO (Global)     │ Inclusivity, no non-therapeutic use in        │
    │                     │ children, full transparency + consent.        │
    │ MIND Act (US prop.) │ FTC regulation of neural data; prohibits      │
    │                     │ selling/exploiting thought-derived data.      │
    │ EU AI Act           │ Neural profiling = high-risk AI use case.     │
    │                     │ Human oversight mandatory.                    │
    │ Brazil AI Bill 2338 │ Risk-based framework; medical AI = high-risk. │
    └─────────────────────┴──────────────────────────────────────────────┘

This module provides:
1. A structured data model for describing neural/behavioral data collection
2. A multi-jurisdiction compliance checker that flags violations
3. A consent adequacy analyzer
4. A data minimization auditor
5. A machine-readable compliance report generator

Author: Cognitive Canary Project
License: MIT
Version: 6.1 (March 2026)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum


# =============================================================================
# ENUMERATIONS
# =============================================================================

class DataCategory(str, Enum):
    """Categories of neural/behavioral data as recognized in 2026 law."""
    CURSOR_MOVEMENT     = "cursor_movement"       # Mouse trajectories
    KEYSTROKE_DYNAMICS  = "keystroke_dynamics"     # Typing timing/pressure
    EEG_RAW             = "eeg_raw"                # Raw EEG waveforms
    EEG_BAND_POWER      = "eeg_band_power"         # Spectral features
    EEG_FINGERPRINT     = "eeg_fingerprint"        # Identity features
    GAZE_TRACKING       = "gaze_tracking"          # Eye movement
    FACIAL_EMG          = "facial_emg"             # Facial muscle signals
    COGNITIVE_STATE     = "cognitive_state"        # Inferred attention/emotion
    NEURODIVERGENCE     = "neurodivergence"        # Inferred ASD/ADHD
    MENTAL_HEALTH       = "mental_health"          # Inferred depression/anxiety
    INTENT              = "intent"                 # Inferred goals/intentions
    CONNECTOME          = "connectome"             # Neural connectivity map


class CollectionContext(str, Enum):
    """Context in which neural data is collected."""
    MEDICAL_CLINICAL    = "medical_clinical"       # Clinical/therapeutic
    RESEARCH_IRB        = "research_irb"           # IRB-approved research
    WORKPLACE           = "workplace"              # Employment context
    EDUCATIONAL         = "educational"            # School/testing
    CONSUMER_PRODUCT    = "consumer_product"       # Commercial product
    ADVERTISING         = "advertising"            # Behavioral advertising
    LAW_ENFORCEMENT     = "law_enforcement"        # Criminal investigation
    MILITARY            = "military"               # Defense applications


class ConsentType(str, Enum):
    """Type of consent provided."""
    NONE                = "none"
    IMPLIED             = "implied"                # Browsing = consent (NOT valid)
    OPT_OUT             = "opt_out"                # Default on, can opt out
    OPT_IN              = "opt_in"                 # Explicit opt-in
    WRITTEN_INFORMED    = "written_informed"       # Written, specific, informed
    ONGOING_REVOCABLE   = "ongoing_revocable"      # Can withdraw anytime


class ViolationSeverity(str, Enum):
    """Severity of a compliance violation."""
    INFO     = "info"      # Informational — best practice gap
    WARNING  = "warning"   # Moderate risk — action recommended
    CRITICAL = "critical"  # High risk — likely illegal in some jurisdictions
    BLOCKED  = "blocked"   # Prohibited — illegal in named jurisdictions


# =============================================================================
# DATA COLLECTION DESCRIPTOR
# =============================================================================

@dataclass
class NeuralDataCollection:
    """
    Describes a neural/behavioral data collection operation.

    Fill this in to describe what data your application (or a third-party
    you integrate with) is collecting, and pass it to NeuroAuditEngine
    for a compliance check.
    """
    # What is being collected
    data_categories: List[DataCategory]

    # Who is collecting and in what context
    collector_name: str
    collection_context: CollectionContext

    # Consent and transparency
    consent_type: ConsentType
    privacy_policy_url: Optional[str] = None
    data_retention_days: Optional[int] = None  # None = indefinite

    # Data flows
    shared_with_third_parties: bool = False
    shared_for_advertising: bool = False
    sold_to_data_brokers: bool = False
    used_for_employment_decisions: bool = False
    used_for_lending_or_insurance: bool = False

    # Subject characteristics
    includes_minors: bool = False
    subjects_have_disabilities: bool = False

    # Technical controls
    data_is_anonymized: bool = False
    anonymization_method: Optional[str] = None
    encryption_at_rest: bool = False
    encryption_in_transit: bool = False

    # Geographic scope
    jurisdictions: List[str] = field(default_factory=lambda: ["US"])


# =============================================================================
# COMPLIANCE VIOLATION
# =============================================================================

@dataclass
class ComplianceViolation:
    """A single compliance violation finding."""
    severity: ViolationSeverity
    jurisdiction: str              # Which law is implicated
    rule: str                      # Short rule identifier
    description: str               # Human-readable description
    remediation: str               # What to do to fix it
    legal_reference: str           # Citation / act / article


# =============================================================================
# COMPLIANCE REPORT
# =============================================================================

@dataclass
class NeuroComplianceReport:
    """Full compliance audit report."""
    collection: NeuralDataCollection
    violations: List[ComplianceViolation]
    overall_risk: ViolationSeverity
    is_compliant: bool
    jurisdictions_checked: List[str]
    summary: str

    def to_json(self) -> str:
        """Serialize report to JSON for regulatory submission."""
        return json.dumps(asdict(self), indent=2, default=str)

    def print_summary(self) -> None:
        """Print a human-readable summary."""
        print(f"\n{'='*65}")
        print(f"NEURO COMPLIANCE AUDIT — {self.collection.collector_name}")
        print(f"{'='*65}")
        print(f"Overall Risk:     {self.overall_risk.value.upper()}")
        print(f"Compliant:        {'✓ YES' if self.is_compliant else '✗ NO'}")
        print(f"Jurisdictions:    {', '.join(self.jurisdictions_checked)}")
        print(f"Violations Found: {len(self.violations)}")
        print(f"\nSummary: {self.summary}")
        print(f"\n{'─'*65}")
        if not self.violations:
            print("  No violations found.")
        for v in sorted(self.violations, key=lambda x: ['info','warning','critical','blocked'].index(x.severity.value), reverse=True):
            icon = {'info': 'ℹ', 'warning': '⚠', 'critical': '✗', 'blocked': '🚫'}[v.severity.value]
            print(f"\n  {icon} [{v.severity.value.upper()}] {v.jurisdiction} — {v.rule}")
            print(f"     {v.description}")
            print(f"     → {v.remediation}")
            print(f"     Ref: {v.legal_reference}")
        print(f"\n{'='*65}")


# =============================================================================
# AUDIT ENGINE
# =============================================================================

class NeuroAuditEngine:
    """
    Multi-jurisdiction neural data compliance checker.

    Checks a NeuralDataCollection descriptor against the 2026 neurorights
    legislative landscape and returns a NeuroComplianceReport.

    Usage
    -----
    >>> collection = NeuralDataCollection(
    ...     data_categories=[DataCategory.EEG_RAW, DataCategory.COGNITIVE_STATE],
    ...     collector_name="WorkplaceHR Inc.",
    ...     collection_context=CollectionContext.WORKPLACE,
    ...     consent_type=ConsentType.IMPLIED,
    ...     used_for_employment_decisions=True,
    ...     jurisdictions=["US", "Chile", "EU"]
    ... )
    >>> engine = NeuroAuditEngine()
    >>> report = engine.audit(collection)
    >>> report.print_summary()
    """

    def audit(self, collection: NeuralDataCollection) -> NeuroComplianceReport:
        """Run a full multi-jurisdiction compliance audit."""
        violations: List[ComplianceViolation] = []

        # Run all jurisdiction checks
        violations.extend(self._check_universal(collection))
        violations.extend(self._check_unesco(collection))
        violations.extend(self._check_chile(collection))
        violations.extend(self._check_colorado(collection))
        violations.extend(self._check_eu_ai_act(collection))
        violations.extend(self._check_mind_act(collection))
        violations.extend(self._check_brazil(collection))

        # Determine overall risk
        if any(v.severity == ViolationSeverity.BLOCKED for v in violations):
            overall_risk = ViolationSeverity.BLOCKED
        elif any(v.severity == ViolationSeverity.CRITICAL for v in violations):
            overall_risk = ViolationSeverity.CRITICAL
        elif any(v.severity == ViolationSeverity.WARNING for v in violations):
            overall_risk = ViolationSeverity.WARNING
        else:
            overall_risk = ViolationSeverity.INFO

        is_compliant = overall_risk not in (
            ViolationSeverity.CRITICAL, ViolationSeverity.BLOCKED
        )

        summary = self._generate_summary(collection, violations, overall_risk)

        return NeuroComplianceReport(
            collection=collection,
            violations=violations,
            overall_risk=overall_risk,
            is_compliant=is_compliant,
            jurisdictions_checked=self._get_checked_jurisdictions(collection),
            summary=summary,
        )

    # -------------------------------------------------------------------------
    # UNIVERSAL CHECKS (apply everywhere)
    # -------------------------------------------------------------------------

    def _check_universal(self, c: NeuralDataCollection) -> List[ComplianceViolation]:
        violations = []

        # No consent at all
        if c.consent_type == ConsentType.NONE:
            violations.append(ComplianceViolation(
                severity=ViolationSeverity.BLOCKED,
                jurisdiction="Universal",
                rule="CONSENT_REQUIRED",
                description="Neural data collection without any consent is prohibited in all major jurisdictions.",
                remediation="Implement explicit opt-in consent before any neural/behavioral data collection begins.",
                legal_reference="UNESCO Neurotechnology Ethics Recommendation (2025), Art. 3; GDPR Art. 7",
            ))

        # Selling neural data to brokers
        if c.sold_to_data_brokers:
            violations.append(ComplianceViolation(
                severity=ViolationSeverity.BLOCKED,
                jurisdiction="Chile / Colorado / MIND Act",
                rule="NO_DATA_BROKER_SALE",
                description="Sale of neural/EEG data to data brokers is prohibited under Chile's constitutional amendment and Colorado law.",
                remediation="Remove all data broker integrations for neural data categories. This applies to cursor dynamics, keystroke patterns, EEG, and inferred cognitive states.",
                legal_reference="Chile Constitutional Amendment Art. 19 (2022); Colorado Privacy Act SB 24-041 (2024)",
            ))

        # No data retention policy
        if c.data_retention_days is None:
            violations.append(ComplianceViolation(
                severity=ViolationSeverity.WARNING,
                jurisdiction="Universal",
                rule="RETENTION_POLICY_REQUIRED",
                description="Indefinite neural data retention creates compounding re-identification risk over time.",
                remediation="Implement a data retention policy. Recommended maximum: 90 days for raw neural signals; 365 days for aggregated features.",
                legal_reference="UNESCO Neurotechnology Ethics Recommendation (2025), § Data Governance",
            ))

        # Inferring neurodivergence without medical context
        if (DataCategory.NEURODIVERGENCE in c.data_categories and
                c.collection_context not in (CollectionContext.MEDICAL_CLINICAL, CollectionContext.RESEARCH_IRB)):
            violations.append(ComplianceViolation(
                severity=ViolationSeverity.CRITICAL,
                jurisdiction="Universal / EU",
                rule="NO_NEURODIVERGENCE_INFERENCE",
                description="Inferring neurodivergence (ADHD, autism spectrum) from behavioral data outside a clinical context constitutes prohibited health discrimination.",
                remediation="Remove neurodivergence inference from non-clinical data pipelines. Classify this as health data and apply medical data protections.",
                legal_reference="ADA (US); EU AI Act Art. 5 (prohibited practices); GDPR Art. 9 (special categories)",
            ))

        return violations

    # -------------------------------------------------------------------------
    # UNESCO RECOMMENDATION (2025)
    # -------------------------------------------------------------------------

    def _check_unesco(self, c: NeuralDataCollection) -> List[ComplianceViolation]:
        violations = []

        # Non-therapeutic use on minors
        if c.includes_minors and c.collection_context not in (
            CollectionContext.MEDICAL_CLINICAL, CollectionContext.RESEARCH_IRB
        ):
            violations.append(ComplianceViolation(
                severity=ViolationSeverity.BLOCKED,
                jurisdiction="UNESCO",
                rule="NO_MINOR_NEURAL_DATA",
                description="UNESCO explicitly warns against non-therapeutic neurotechnology use on children. Collecting behavioral/EEG data from minors in commercial/educational contexts without therapeutic justification is prohibited.",
                remediation="Exclude minors from neural data collection pipelines. If educational context is unavoidable, obtain separate IRB-equivalent review and parental written consent.",
                legal_reference="UNESCO Recommendation on the Ethics of Neurotechnology (2025), Principle 4",
            ))

        # No transparency / privacy policy
        if c.privacy_policy_url is None:
            violations.append(ComplianceViolation(
                severity=ViolationSeverity.WARNING,
                jurisdiction="UNESCO",
                rule="TRANSPARENCY_REQUIRED",
                description="UNESCO requires full transparency about what neural data is collected and how it is used.",
                remediation="Publish a specific neural data privacy policy explaining: data categories collected, purpose, retention period, third-party sharing, and user rights.",
                legal_reference="UNESCO Recommendation on the Ethics of Neurotechnology (2025), Principle 3 (Transparency)",
            ))

        return violations

    # -------------------------------------------------------------------------
    # CHILE (World's first constitutional neurorights)
    # -------------------------------------------------------------------------

    def _check_chile(self, c: NeuralDataCollection) -> List[ComplianceViolation]:
        if "Chile" not in c.jurisdictions:
            return []

        violations = []

        # Collecting EEG or fingerprint data in Chile
        high_risk_categories = {DataCategory.EEG_RAW, DataCategory.EEG_FINGERPRINT,
                                  DataCategory.CONNECTOME, DataCategory.MENTAL_HEALTH,
                                  DataCategory.NEURODIVERGENCE}
        collected_high_risk = high_risk_categories & set(c.data_categories)

        if collected_high_risk and c.consent_type not in (
            ConsentType.WRITTEN_INFORMED, ConsentType.ONGOING_REVOCABLE
        ):
            violations.append(ComplianceViolation(
                severity=ViolationSeverity.BLOCKED,
                jurisdiction="Chile",
                rule="CHILE_NEURAL_CONSENT",
                description=f"Chile's constitutional amendment requires explicit written consent for collection of: {', '.join(d.value for d in collected_high_risk)}. Implied or opt-out consent is insufficient.",
                remediation="Obtain written informed consent with specific disclosure of data categories, purposes, and retention. Consent must be individually revocable.",
                legal_reference="Chilean Constitution Art. 19 (as amended 2022); Supreme Court Emotiv ruling (2023)",
            ))

        # Use for advertising in Chile
        if c.shared_for_advertising and any(
            d in c.data_categories for d in [DataCategory.EEG_RAW, DataCategory.COGNITIVE_STATE,
                                               DataCategory.MENTAL_HEALTH, DataCategory.INTENT]
        ):
            violations.append(ComplianceViolation(
                severity=ViolationSeverity.BLOCKED,
                jurisdiction="Chile",
                rule="CHILE_NO_NEURAL_ADVERTISING",
                description="Using neural/cognitive state data for advertising purposes in Chile violates the constitutional right to mental privacy and free will.",
                remediation="Remove neural signal features from advertising pipelines entirely. This includes inferred emotion, attention scores, and intent signals.",
                legal_reference="Chilean Constitution Art. 19, Nº 1 (2022 amendment)",
            ))

        return violations

    # -------------------------------------------------------------------------
    # COLORADO PRIVACY ACT (Neural data provisions)
    # -------------------------------------------------------------------------

    def _check_colorado(self, c: NeuralDataCollection) -> List[ComplianceViolation]:
        if "US" not in c.jurisdictions and "Colorado" not in c.jurisdictions:
            return []

        violations = []

        neural_categories = {DataCategory.EEG_RAW, DataCategory.EEG_BAND_POWER,
                              DataCategory.EEG_FINGERPRINT, DataCategory.COGNITIVE_STATE,
                              DataCategory.KEYSTROKE_DYNAMICS}
        collected_neural = neural_categories & set(c.data_categories)

        if collected_neural and c.consent_type in (ConsentType.NONE, ConsentType.IMPLIED):
            violations.append(ComplianceViolation(
                severity=ViolationSeverity.CRITICAL,
                jurisdiction="Colorado (USA)",
                rule="COLORADO_NEURAL_SENSITIVE",
                description=f"Colorado Privacy Act classifies {', '.join(d.value for d in collected_neural)} as 'sensitive personal data' requiring explicit opt-in consent.",
                remediation="Implement an explicit opt-in consent flow before any neural data collection. Provide a clear opt-out mechanism that immediately halts collection.",
                legal_reference="Colorado Privacy Act SB 24-041 (2024); CRS § 6-1-1303",
            ))

        return violations

    # -------------------------------------------------------------------------
    # EU AI ACT
    # -------------------------------------------------------------------------

    def _check_eu_ai_act(self, c: NeuralDataCollection) -> List[ComplianceViolation]:
        if "EU" not in c.jurisdictions:
            return []

        violations = []

        # Real-time biometric profiling from neural data (prohibited in public spaces)
        if (c.collection_context == CollectionContext.ADVERTISING and
                any(d in c.data_categories for d in [DataCategory.COGNITIVE_STATE,
                                                        DataCategory.EEG_RAW,
                                                        DataCategory.GAZE_TRACKING])):
            violations.append(ComplianceViolation(
                severity=ViolationSeverity.BLOCKED,
                jurisdiction="EU",
                rule="EU_AI_ACT_PROHIBITED_BIOMETRIC",
                description="Real-time biometric profiling using neural or gaze data for targeted advertising is a prohibited AI practice under the EU AI Act.",
                remediation="Remove all neural/gaze data signals from advertising targeting systems. Document removal and maintain an audit log.",
                legal_reference="EU AI Act Art. 5(1)(b) — prohibited biometric categorization (2024)",
            ))

        # Employment decisions without human oversight
        if (c.used_for_employment_decisions and
                any(d in c.data_categories for d in [DataCategory.COGNITIVE_STATE,
                                                       DataCategory.KEYSTROKE_DYNAMICS,
                                                       DataCategory.EEG_BAND_POWER])):
            violations.append(ComplianceViolation(
                severity=ViolationSeverity.CRITICAL,
                jurisdiction="EU",
                rule="EU_AI_ACT_HIGH_RISK_EMPLOYMENT",
                description="Using neural/behavioral data for employment decisions is a high-risk AI use case under the EU AI Act, requiring mandatory human oversight, conformity assessment, and registration.",
                remediation="Implement mandatory human review for all employment decisions influenced by behavioral data. Register the AI system and conduct a conformity assessment.",
                legal_reference="EU AI Act Annex III (high-risk AI systems), Art. 14 (human oversight)",
            ))

        return violations

    # -------------------------------------------------------------------------
    # US MIND ACT (proposed 2025)
    # -------------------------------------------------------------------------

    def _check_mind_act(self, c: NeuralDataCollection) -> List[ComplianceViolation]:
        if "US" not in c.jurisdictions:
            return []

        violations = []

        # Neural data for advertising (MIND Act target)
        if (c.shared_for_advertising and
                any(d in c.data_categories for d in [DataCategory.EEG_RAW,
                                                       DataCategory.EEG_BAND_POWER,
                                                       DataCategory.COGNITIVE_STATE,
                                                       DataCategory.INTENT])):
            violations.append(ComplianceViolation(
                severity=ViolationSeverity.WARNING,  # Proposed, not yet law
                jurisdiction="US (MIND Act — proposed)",
                rule="MIND_ACT_NEURAL_ADVERTISING",
                description="The proposed MIND Act would prohibit exploitation of neural data (including inferred intent and cognitive state) for advertising. While not yet enacted, compliance now prevents future liability.",
                remediation="Remove neural inference signals from advertising pipelines now. Document removal as evidence of proactive compliance.",
                legal_reference="MIND Act (Management of Individuals' Neural Data Act, proposed 2025), Senators Schumer/Cantwell/Markey",
            ))

        return violations

    # -------------------------------------------------------------------------
    # BRAZIL AI BILL 2338
    # -------------------------------------------------------------------------

    def _check_brazil(self, c: NeuralDataCollection) -> List[ComplianceViolation]:
        if "Brazil" not in c.jurisdictions:
            return []

        violations = []

        # High-risk AI without risk assessment
        high_risk_contexts = {CollectionContext.MEDICAL_CLINICAL, CollectionContext.EDUCATIONAL,
                               CollectionContext.LAW_ENFORCEMENT}
        if (c.collection_context in high_risk_contexts and
                any(d in c.data_categories for d in [DataCategory.EEG_RAW,
                                                       DataCategory.COGNITIVE_STATE,
                                                       DataCategory.MENTAL_HEALTH])):
            violations.append(ComplianceViolation(
                severity=ViolationSeverity.CRITICAL,
                jurisdiction="Brazil",
                rule="BRAZIL_HIGH_RISK_ASSESSMENT",
                description="Brazil's AI Bill 2338 classifies medical, educational, and law enforcement AI systems using neural data as 'high-risk', requiring a mandatory impact assessment and ANPD notification.",
                remediation="Conduct an AI Impact Assessment (AIA) for this system. Register with ANPD (Brazil's data protection authority). Implement human oversight mechanisms.",
                legal_reference="Brazil AI Bill 2338/2023, Art. 17 (high-risk systems), ANPD oversight",
            ))

        return violations

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _get_checked_jurisdictions(self, c: NeuralDataCollection) -> List[str]:
        checked = ["Universal", "UNESCO"]
        if "Chile" in c.jurisdictions:
            checked.append("Chile")
        if "US" in c.jurisdictions or "Colorado" in c.jurisdictions:
            checked.append("Colorado (USA)")
            checked.append("US (MIND Act)")
        if "EU" in c.jurisdictions:
            checked.append("EU (AI Act)")
        if "Brazil" in c.jurisdictions:
            checked.append("Brazil (AI Bill 2338)")
        return checked

    def _generate_summary(
        self,
        c: NeuralDataCollection,
        violations: List[ComplianceViolation],
        risk: ViolationSeverity,
    ) -> str:
        n_blocked = sum(1 for v in violations if v.severity == ViolationSeverity.BLOCKED)
        n_critical = sum(1 for v in violations if v.severity == ViolationSeverity.CRITICAL)
        n_warning = sum(1 for v in violations if v.severity == ViolationSeverity.WARNING)

        if risk == ViolationSeverity.BLOCKED:
            return (
                f"PROHIBITED: {n_blocked} illegal collection practice(s) detected. "
                f"Immediate remediation required before any deployment. "
                f"This configuration violates neurorights law in: "
                f"{', '.join(set(v.jurisdiction for v in violations if v.severity == ViolationSeverity.BLOCKED))}."
            )
        elif risk == ViolationSeverity.CRITICAL:
            return (
                f"HIGH RISK: {n_critical} critical violation(s) require immediate attention. "
                f"Current configuration likely violates applicable law in named jurisdictions."
            )
        elif risk == ViolationSeverity.WARNING:
            return (
                f"MODERATE RISK: {n_warning} best-practice gaps identified. "
                f"Compliant in current law but exposed to emerging regulations."
            )
        else:
            return "LOW RISK: Collection appears compliant across all checked jurisdictions."


# =============================================================================
# EXAMPLE / VALIDATION
# =============================================================================

if __name__ == "__main__":
    engine = NeuroAuditEngine()

    print("\n" + "="*65)
    print("SCENARIO 1: Workplace 'Productivity AI' — Flagrant Violations")
    print("="*65)
    bad_collection = NeuralDataCollection(
        data_categories=[
            DataCategory.KEYSTROKE_DYNAMICS,
            DataCategory.COGNITIVE_STATE,
            DataCategory.EEG_BAND_POWER,
            DataCategory.NEURODIVERGENCE,
        ],
        collector_name="WorkforceAI Corp",
        collection_context=CollectionContext.WORKPLACE,
        consent_type=ConsentType.IMPLIED,
        shared_for_advertising=True,
        sold_to_data_brokers=True,
        used_for_employment_decisions=True,
        includes_minors=False,
        jurisdictions=["US", "Chile", "EU"],
    )
    report1 = engine.audit(bad_collection)
    report1.print_summary()

    print("\n" + "="*65)
    print("SCENARIO 2: Compliant Clinical BCI Research")
    print("="*65)
    good_collection = NeuralDataCollection(
        data_categories=[
            DataCategory.EEG_RAW,
            DataCategory.EEG_BAND_POWER,
        ],
        collector_name="NeuroRehab Research Institute",
        collection_context=CollectionContext.RESEARCH_IRB,
        consent_type=ConsentType.WRITTEN_INFORMED,
        privacy_policy_url="https://example.edu/neural-privacy-policy",
        data_retention_days=365,
        shared_with_third_parties=False,
        shared_for_advertising=False,
        sold_to_data_brokers=False,
        used_for_employment_decisions=False,
        includes_minors=False,
        encryption_at_rest=True,
        encryption_in_transit=True,
        jurisdictions=["US"],
    )
    report2 = engine.audit(good_collection)
    report2.print_summary()
