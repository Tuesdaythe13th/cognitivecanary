# ☢️ Cognitive Canary: Active Defense Against Neural Inference

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fm4-aQkAzqazirgdhQ6OVCtR8HQXwTyq#scrollTo=eqpOD7_c_STz)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Prototype](https://img.shields.io/badge/Status-Prototype_v5.0-red.svg)]()
[![Theme: d/acc](https://img.shields.io/badge/Theme-d%2Facc-00ff00.svg)]()

> **"We didn't hack the password. We hacked the inference."**

---

## 🚨 The Problem: The Inference Gap
As AI systems transition from passive tools to active agents, they are developing the capacity to infer sensitive cognitive states—**intent, neurodivergence, emotion, and fatigue**—from behavioral metadata alone (e.g., cursor micro-tremors, keystroke flight time). 

This data is currently unprotected by law. While encryption protects *what you say*, nothing protects *how you move*.

## 🛡️ The Solution: Cognitive Canary
**Cognitive Canary** is a d/acc (Defensive Acceleration) protocol that automates **Adversarial Camouflage**. It injects high-entropy, mathematically generated noise into user telemetry to degrade the utility of covertly harvested behavioral data.

This repository contains **v5.0 of the Research Notebook**, a fully executable proof-of-concept that:
1.  **Ingests Real Human Data** (from the MouseDynamics challenge).
2.  **Trains a Surveillance AI** to detect non-human behavior.
3.  **Generates "Poison"** using Lissajous curves masked with physiological jitter.
4.  **Breaks the AI** (>95% Bypass Rate) by forcing it into a state of high-confidence misclassification.

---

## ⚡ Quick Start (Run the Demo)

The easiest way to reproduce the results is to run the Jupyter Notebook in Google Colab.

1.  Click the **"Open in Colab"** badge above.
2.  Run **Step [00]** to **Step [11]** sequentially.
3.  Watch the **Kinetic Replay** to see the adversarial camouflage in action.
4.  Download the generated **Zero-Knowledge Circuit** (`neuro_shield.circom`) from the artifacts folder.

---

## 🧠 Theory of Operation

### 1. The Poison Engine (Gradient Starvation)
Surveillance models are lazy. They look for the easiest feature to classify. Cognitive Canary exploits this by generating **Lissajous Curves**—mathematically perfect loops—and masking them with **Uniform noise** calibrated to mimic Human Physiological Tremor.

The surveillance model learns to classify the "smoothness" of the curve as human, ignoring the underlying lack of semantic intent. This is known as **Gradient Starvation**.

### 2. Zero-Neuro (The Future Architecture)
The notebook also generates `neuro_shield.circom`. This is a template for a **Zero-Knowledge Proof (ZKP)**. In the future, instead of streaming raw mouse data to prove you are human (leaking your cognitive state), you will generate a proof locally and send only the cryptographic hash.

> **Privacy by Policy** = "Please don't look at my data."  
> **Privacy by Engineering** = "You can look, but you will see only math."

---

## 📊 Results (From v5.0 Run)

| Metric | Result | Meaning |
| :--- | :--- | :--- |
| **Baseline Accuracy** | **98.4%** | The Surveillance AI is highly effective at spotting standard bots. |
| **Canary Bypass Rate** | **96.5%** | The AI thought our "Poison" was human 96.5% of the time. |
| **Entropy State** | **Low (<0.06)** | The AI was **confidently wrong** (Worst-case failure mode). |
| **Dataset Provenance** | **SHA-256** | The ZK circuit is cryptographically stamped with the training data hash. |

---

## 📂 Repository Structure

```text
├── aritfex_apart_defense_CC.ipynb   # The Main Research Kernel (Run this)
├── README.md                        # This file
└── (Generated at runtime to Google Drive):
    ├── neuro_shield.circom          # Generated ZK-Circuit
    ├── surveillance_model.joblib    # The trained adversary (for testing)
    ├── scaler.joblib                # Normalization parameters
    └── README.md                    # Generated Manifesto
