# Contributing to Cognitive Canary

Thank you for your interest in contributing to Cognitive Canary! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Guidelines](#contribution-guidelines)
- [Code Style](#code-style)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Security & Ethics](#security--ethics)
- [License](#license)

---

## Code of Conduct

### Our Values

- **Privacy First**: We build tools that protect individual cognitive freedom
- **Transparency**: All code is open-source and thoroughly documented
- **Evidence-Based**: Claims must be backed by reproducible experiments
- **Ethical Use**: We prevent misuse while enabling legitimate defense
- **Inclusivity**: We welcome contributors from all backgrounds

### Expected Behavior

- ✅ Be respectful and constructive in discussions
- ✅ Provide evidence and citations for technical claims
- ✅ Document your code thoroughly
- ✅ Test your contributions before submitting
- ✅ Respect the project's ethical guidelines (see README)

### Unacceptable Behavior

- ❌ Submitting code designed for offensive/malicious use
- ❌ Bypassing safety checks or productivity failsafes
- ❌ Removing ethical guidelines or deployment checklists
- ❌ Harassing other contributors
- ❌ Submitting untested or undocumented code

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- NumPy, SciPy (see `requirements.txt`)
- Familiarity with signal processing and machine learning concepts
- Understanding of the project's [ethical framework](README.md#legal--ethical-framework)

### Understanding the Codebase

Before contributing, familiarize yourself with:

1. **Core Modules**: Read the docstrings in each `.py` file
2. **Architecture**: Review the system diagram in README
3. **Notebooks**: Run the Colab demo to understand the workflow
4. **Constants**: Check `constants.py` for configurable parameters

**Key Files:**
- `lissajous_3d.py` - 3D cursor obfuscation engine
- `adaptive_tremor.py` - Tremor learning and injection
- `keystroke_jitter.py` - Keystroke dynamics obfuscation
- `task_classifier_v2.py` - Task detection and productivity monitoring
- `gradient_auditor.py` - ML attack detection
- `spectral_canary.py` - EEG frequency injection
- `spectral_utils.py` - Shared spectral analysis utilities
- `noise_generators.py` - Shared noise generation utilities
- `constants.py` - Global configuration constants

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/cognitivecanary.git
cd cognitivecanary
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest pytest-cov black flake8 mypy
```

### 3. Verify Installation

```bash
# Run example usage blocks to verify setup
python lissajous_3d.py
python adaptive_tremor.py
python gradient_auditor.py
```

### 4. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

---

## Contribution Guidelines

### Types of Contributions

We welcome:

✅ **Bug Fixes**: Fix issues in existing code
✅ **Performance Improvements**: Optimize hot paths, reduce latency
✅ **Documentation**: Improve docstrings, add examples, fix typos
✅ **New Obfuscation Engines**: Add novel defense mechanisms
✅ **Attack Detection**: Enhance gradient auditor capabilities
✅ **Test Coverage**: Add unit tests and integration tests
✅ **Platform Support**: iOS, Android, browser extensions
✅ **Reproducibility**: Improve dataset documentation, add benchmarks

⚠️ **Requires Discussion First:**
- Major architectural changes
- New dependencies (especially heavy ML frameworks)
- Changes to ethical guidelines or deployment checklist
- Removal of safety features (productivity failsafe, etc.)

❌ **Not Accepted:**
- Code designed for offensive surveillance
- Features that bypass user consent mechanisms
- Removal of biomimicry validation or plausibility checks
- Code without documentation or tests

---

## Code Style

### Python Style Guide

We follow **PEP 8** with some project-specific conventions:

```python
# ✅ GOOD: Descriptive names, type hints, docstrings
def compute_spectral_entropy(signal: np.ndarray,
                            fs: float = 100.0) -> float:
    """
    Compute spectral entropy using Welch's method.

    Parameters
    ----------
    signal : np.ndarray
        1D time-series signal
    fs : float, default=100.0
        Sampling frequency in Hz

    Returns
    -------
    float
        Spectral entropy in nats
    """
    # Implementation with inline comments for complex logic
    frequencies, psd = welch(signal, fs=fs, nperseg=256)
    psd_normalized = psd / (np.sum(psd) + 1e-12)  # Avoid division by zero
    return float(entropy(psd_normalized, base=np.e))

# ❌ BAD: No types, no docstring, unclear names
def calc(x, y):
    a, b = welch(x, fs=y, nperseg=256)
    c = b / np.sum(b)
    return entropy(c, base=np.e)
```

### Key Conventions

1. **Type Hints**: All function signatures must include type hints
2. **Docstrings**: Use NumPy-style docstrings with Parameters/Returns sections
3. **Constants**: Use `constants.py` instead of magic numbers
4. **Naming**:
   - Classes: `PascalCase` (e.g., `LissajousEngine`)
   - Functions: `snake_case` (e.g., `compute_spectral_entropy`)
   - Constants: `UPPER_SNAKE_CASE` (e.g., `SPECTRAL_ENTROPY_TARGET`)
   - Private methods: `_leading_underscore` (e.g., `_normalize_signal`)
5. **Line Length**: Max 100 characters (slight relaxation from PEP 8's 79)
6. **Imports**: Group in order: stdlib, third-party, local modules

### Formatting Tools

```bash
# Auto-format code (recommended before commit)
black --line-length 100 your_file.py

# Check style compliance
flake8 --max-line-length=100 your_file.py

# Type checking
mypy your_file.py
```

---

## Testing Requirements

### Test Coverage

All new features **must** include tests. We use `pytest`.

**Minimum Requirements:**
- Unit tests for all public functions/methods
- Integration tests for end-to-end workflows
- Example usage blocks in `if __name__ == "__main__":`

### Writing Tests

Create tests in a `tests/` directory (to be added):

```python
# tests/test_spectral_utils.py
import numpy as np
from spectral_utils import compute_spectral_entropy

def test_spectral_entropy_white_noise():
    """White noise should have high spectral entropy."""
    signal = np.random.randn(1000)
    entropy = compute_spectral_entropy(signal, fs=100.0)
    assert entropy > 3.0, "White noise entropy unexpectedly low"

def test_spectral_entropy_sine_wave():
    """Pure sine wave should have low spectral entropy."""
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t)
    entropy = compute_spectral_entropy(signal, fs=1000.0)
    assert entropy < 2.0, "Sine wave entropy unexpectedly high"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_spectral_utils.py
```

### Validation Checklist

Before submitting, verify:

- [ ] All new functions have type hints
- [ ] All new functions have NumPy-style docstrings
- [ ] Unit tests added with >80% coverage
- [ ] Example usage block runs without errors
- [ ] Code passes `flake8` style checks
- [ ] Code passes `mypy` type checks (if applicable)
- [ ] Biomimicry validation passes (spectral entropy, SNR, etc.)

---

## Pull Request Process

### 1. Pre-Submission Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] Documentation updated (README, docstrings, etc.)
- [ ] No sensitive data or credentials committed
- [ ] Ethical guidelines respected (no offensive capabilities)
- [ ] Changelog entry added (if applicable)

### 2. Commit Messages

Use descriptive commit messages:

```bash
# ✅ GOOD
git commit -m "feat: Add Halo2 zero-knowledge verifier for entropy bounds"
git commit -m "fix: Correct pink noise generation in keystroke jitter (#42)"
git commit -m "docs: Improve spectral entropy docstring with examples"
git commit -m "test: Add unit tests for 3D Lissajous coverage validation"

# ❌ BAD
git commit -m "stuff"
git commit -m "fix bug"
git commit -m "update"
```

**Format:** `<type>: <description>`

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Add/update tests
- `refactor`: Code restructuring without behavior change
- `perf`: Performance improvement
- `chore`: Maintenance (dependencies, build, etc.)

### 3. Creating a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request on GitHub with:
   - **Title**: Clear, descriptive summary
   - **Description**:
     - What problem does this solve?
     - How does it work?
     - Evidence/benchmarks (if applicable)
     - Screenshots (if UI changes)
   - **Testing**: Describe how you tested the changes
   - **Ethical Review**: Confirm compliance with ethical guidelines

3. Link related issues (e.g., "Closes #42")

### 4. Review Process

- Maintainers will review within 1 week
- Address feedback by pushing new commits to your branch
- Once approved, maintainers will merge (squash merge preferred)
- Your contribution will be acknowledged in CHANGELOG and README

### 5. After Merge

- Delete your feature branch
- Pull latest changes from main
- Update your fork

---

## Security & Ethics

### Responsible Disclosure

If you discover a security vulnerability or ethical issue:

1. **DO NOT** open a public issue
2. Email maintainers privately: [INSERT CONTACT EMAIL]
3. Provide: Description, reproduction steps, potential impact
4. Allow 90 days for fix before public disclosure

### Ethical Guidelines

**Contributions must:**
- ✅ Protect user privacy and cognitive freedom
- ✅ Include productivity failsafe mechanisms
- ✅ Preserve biomimicry validation checks
- ✅ Document appropriate use cases

**Contributions must NOT:**
- ❌ Enable surveillance or profiling
- ❌ Bypass user consent or awareness
- ❌ Cause harm to system performance
- ❌ Violate laws or terms of service

### Dual-Use Considerations

Cognitive Canary is defensive technology. Features should:
- Default to **user protection** (not evasion for malicious purposes)
- Include **transparency mechanisms** (logging, monitoring)
- Respect **appropriate use contexts** (defined in README)

---

## License

By contributing, you agree that your contributions will be licensed under the **MIT License** (same as the project).

You retain copyright to your contributions, but grant the project permission to use, modify, and distribute them under MIT terms.

---

## Questions?

- **General Questions**: Open a GitHub Discussion
- **Bug Reports**: Open a GitHub Issue (use template)
- **Feature Requests**: Open a GitHub Issue (describe use case, evidence)
- **Security Issues**: Email maintainers privately

---

## Acknowledgments

We value all contributions, from code to documentation to bug reports. Contributors will be acknowledged in:
- README acknowledgments section
- CHANGELOG for each release
- GitHub contributors list

Thank you for helping build privacy-preserving technology! 🛡️

---

**Cognitive Canary Project**
*Active Defense Against Neural Inference*
MIT License | 2025
