# AI Risk Manager — Cost-Aware Payment Fraud Detection

An AI-powered payment fraud detection and risk scoring system that evaluates payment transactions, estimates fraud risk, and recommends whether a transaction should be **allowed or manually reviewed**.

The system combines machine learning, cost-sensitive threshold optimization, risk scoring, and feature-based explanations in an interactive Streamlit dashboard.

---

## 🚀 Project Overview

The system analyzes payment transaction features and produces:

- Fraud probability
- Risk score (0–100)
- Risk level
- Recommended action: `ALLOW` or `REVIEW`
- Feature-based explanation
- Cost-aware decision threshold

The project focuses on **defensive fraud detection and merchant loss reduction**.

---

## 🏗️ System Architecture

```text
Payment Transaction
        ↓
Feature Processing
        ↓
Machine Learning Model
        ↓
Fraud Probability
        ↓
Risk Score (0–100)
        ↓
Cost-Aware Threshold
        ↓
ALLOW / REVIEW
        ↓
Risk Explanation
