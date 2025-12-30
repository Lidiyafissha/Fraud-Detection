# Project: Multi-Platform Fraud Detection System

**Developed for Adey Innovations Inc.**

## 📌 Overview

At **Adey Innovations Inc.**, securing financial transactions is at the core of our mission. This project delivers a robust, machine-learning-driven fraud detection system designed to protect both e-commerce platforms and banking institutions.

By integrating geolocation analysis, transaction velocity patterns, and advanced ensemble modeling, we provide a solution that minimizes financial loss (False Negatives) while preserving a seamless user experience (minimizing False Positives).

---

## 🏢 Business Objective

The financial technology sector faces a constant battle against evolving fraud tactics. Our goal is to:

- **Enhance Security:** Detect sophisticated fraud patterns in real-time.
- **Balance UX & Risk:** Optimize the trade-off between strict security and customer friction.
- **Build Trust:** Provide transparent, explainable AI insights to stakeholders and financial partners.

---

## 🛠️ Technical Workflow

### 1. Data Analysis & Geolocation Intelligence

We process two distinct streams: e-commerce logs and bank credit transactions.

- **IP Intelligence:** We map transaction IP addresses to physical locations to identify high-risk geographic anomalies.
- **Feature Engineering:** Beyond raw data, we calculate **Transaction Velocity** (the speed of repeated purchases) and **Account Aging** (time elapsed since signup) to flag bot-like behavior.

### 2. Predictive Modeling & Evaluation

Since fraudulent transactions are rare compared to legitimate ones, we utilize specialized techniques to prevent the model from being "blinded" by the majority class.

- **Handling Imbalance:** We utilize **SMOTE** (Synthetic Minority Over-sampling Technique) to create a balanced learning environment.
- **Ensemble Power:** We deploy **XGBoost** and **Random Forest** models, which are superior at capturing the complex, non-linear relationships typical of financial crimes.
- **Precision-Recall Focus:** We evaluate success using the **Precision-Recall Curve (AUC-PR)** and **F1-Score**, ensuring we prioritize the detection of fraud without overwhelming the system with false alarms.

### 3. Explainability & Actionable Insights

A "black box" model is not enough for the fintech industry. We use **SHAP (SHapley Additive exPlanations)** to break down why a specific transaction was flagged.

- **Global Drivers:** Identify what factors most consistently indicate fraud across the company.
- **Individual Case Studies:** Analyze "False Positives" to refine rules and reduce customer annoyance.
- **Business Rules:** Translate model data into simple logic, such as: _"Flag all accounts where the purchase occurs within 10 minutes of creation from a high-risk IP range."_

---

## 📈 Impact

- **Reduced Financial Leakage:** Early detection of "burn accounts" prevents immediate loss.
- **Operational Efficiency:** Automated screening allows security teams to focus on high-probability cases.
- **Scalability:** The framework is designed to adapt to new datasets as Adey Innovations expands its financial reach.

## 🧠 Task 2 — Model Building, Training & Evaluation

### Objective

Design, train, and evaluate **robust classification models** capable of detecting fraudulent transactions in **highly imbalanced datasets**, while ensuring **reproducibility, modularity, and fair model comparison**.

This task focuses on **modeling rigor**, **evaluation correctness**, and **engineering best practices**.

---

## 📊 1. Data Preparation

### Dataset Separation

Two datasets are modeled **independently** due to different feature spaces and business contexts:

- **E-commerce transactions**

  - Target: `class`
  - Behavioral + geolocation features

- **Credit card transactions**

  - Target: `Class`
  - PCA-transformed numerical features (V1–V28)

> The datasets are **not merged** to avoid feature leakage and semantic mismatch.

---

### Stratified Train–Test Split

To preserve the extreme fraud imbalance:

- **Stratified splitting** ensures class distribution consistency
- Prevents misleading performance inflation

```text
Train/Test split: 80% / 20%
Stratification key: fraud label
```

---

## 🧩 2. Modular Training Pipeline

All modeling logic is implemented in the **`src/` directory** to ensure:

- Reusability
- Testability
- CI/CD compatibility
- Clean notebook execution

### Core Pipeline Components

| Module                 | Responsibility                 |
| ---------------------- | ------------------------------ |
| `preprocessing.py`     | Feature preparation & encoding |
| `split.py`             | Stratified train-test split    |
| `baseline_model.py`    | Logistic Regression definition |
| `ensemble_model.py`    | Random Forest model            |
| `training_pipeline.py` | Unified training & evaluation  |

---

## 📉 3. Baseline Model — Logistic Regression

### Why Logistic Regression?

- Highly interpretable
- Establishes a **performance floor**
- Ideal for business explanation and risk justification

### Configuration

- Class weighting to account for imbalance
- Regularization to reduce overfitting

### Metrics Used

- **F1-Score** → Balance between Precision & Recall
- **PR-AUC** → Preferred over ROC-AUC for rare events
- **Confusion Matrix** → Operational error analysis

---

## 🌲 4. Ensemble Model — Random Forest

### Why Random Forest?

- Handles non-linear interactions
- Robust to noise
- Strong performance on tabular fraud data

### Tuned Hyperparameters

- `n_estimators`
- `max_depth`
- `min_samples_split`

### Strengths

- Captures behavioral patterns
- Learns complex fraud signatures
- Resistant to overfitting compared to single trees

---

## ⚖️ 5. Evaluation Strategy (Imbalanced Learning)

Fraud detection is **not an accuracy problem**.

### Chosen Metrics

| Metric               | Justification                               |
| -------------------- | ------------------------------------------- |
| **F1-Score**         | Penalizes false positives & false negatives |
| **PR-AUC**           | Focuses on minority (fraud) class           |
| **Confusion Matrix** | Business-impact clarity                     |

> ROC-AUC is avoided as it can be misleading under extreme imbalance.

---

## 🔍 6. Cross-Dataset Model Comparison

Models are trained and evaluated **separately** on:

- Fraud (E-commerce) dataset
- Credit Card dataset

### Observed Patterns

- Random Forest significantly outperforms Logistic Regression
- Credit card data benefits from PCA-engineered features
- E-commerce data benefits from behavioral + geolocation features

---

## 🧪 7. Validation & Testing

To ensure robustness:

- **Unit tests** validate pipeline execution
- Metrics are sanity-checked (0 ≤ score ≤ 1)
- Training is deterministic via fixed random seeds

```bash
pytest
```

---

## 🏆 8. Model Selection & Justification

| Model               | Selection Reason                         |
| ------------------- | ---------------------------------------- |
| Logistic Regression | Interpretability & baseline benchmarking |
| Random Forest       | Best overall fraud detection performance |

**Final choice:**
✔ **Random Forest** for production detection
✔ **Logistic Regression** for explainability & audits

---

## 🎯 Business Value Delivered

- ✔ Reduced false negatives → lower financial loss
- ✔ Controlled false positives → better user experience
- ✔ Scalable pipeline → multi-platform deployment
- ✔ Reproducible experiments → governance & compliance

---

## 🔍 Task 3 — Model Explainability & SHAP Analysis

### Objective

Implement **SHAP (SHapley Additive exPlanations)** analysis to provide **transparent, interpretable insights** into model predictions, enabling stakeholders to understand **why** transactions are flagged as fraudulent and identify the **most critical fraud indicators**.

This task focuses on **explainability**, **feature importance analysis**, and **actionable business insights**.

---

## 🧩 1. SHAP Implementation Architecture

### Core Components

| Component                   | Purpose                               |
| --------------------------- | ------------------------------------- |
| `shap_analysis.py`          | Loads model, generates SHAP explainer |
| `TreeExplainer`             | SHAP explainer for tree-based models  |
| `shap-explainability.ipynb` | Interactive analysis & visualization  |

### Workflow

1. **Model Loading:** Load trained Random Forest model and test data
2. **SHAP Explainer:** Initialize TreeExplainer for Random Forest
3. **SHAP Values:** Compute SHAP values for test set (sampled to 1000 instances)
4. **Visualization:** Generate global and local explanations

---

## 📊 2. Global Feature Importance Analysis

### Top 5 Fraud Drivers

| Rank | Feature                  | Importance | Interpretation                                     |
| ---- | ------------------------ | ---------- | -------------------------------------------------- |
| 1    | `device_id_freq`         | ~0.51      | Device reuse frequency — strongest fraud indicator |
| 2    | `time_since_signup`      | ~0.37      | Account age — new accounts are high-risk           |
| 3    | `ip_int`                 | ~0.02      | IP address encoding                                |
| 4    | `user_id`                | ~0.02      | User identifier patterns                           |
| 5    | `upper_bound_ip_address` | ~0.02      | IP range boundaries                                |

### Key Observations

- **Behavioral Dominance:** `device_id_freq` and `time_since_signup` account for **~88%** of predictive power
- **Static Attributes:** IP range, purchase value, time of day, country frequency, and demographics have **comparatively minor influence**
- **Model Focus:** The model primarily relies on **behavioral patterns** rather than static attributes

---

## 🎯 3. Individual Prediction Analysis

### Case Study Approach

SHAP force plots were generated for three critical prediction types:

1. **True Positive (TP):** Correctly identified fraud

   - Analyze which features pushed prediction toward fraud class
   - Validate model reasoning aligns with business logic

2. **False Positive (FP):** Legitimate transaction flagged incorrectly

   - Identify features causing over-aggressive flagging
   - Refine rules to reduce customer friction

3. **False Negative (FN):** Missed fraud case
   - Understand why model failed to detect fraud
   - Improve detection sensitivity

### Business Impact

- **Operational Efficiency:** Security teams can focus on high-probability cases
- **Customer Experience:** Understanding FP cases helps reduce false alarms
- **Model Refinement:** FN analysis guides feature engineering improvements

---

## 🔬 4. Feature Importance Comparison

### SHAP vs. Built-in Feature Importance

Both methods consistently identify the same top drivers:

- **Device Frequency** dominates both rankings
- **Time Since Signup** is the second most important feature
- **Consistency:** Validates that SHAP explanations align with model internals

### Why This Matters

- **Trust:** Stakeholders can verify model reasoning matches expectations
- **Debugging:** Discrepancies between methods signal potential issues
- **Transparency:** Provides multiple perspectives on feature importance

---

## 💡 5. Counterintuitive Findings

### Geography Hypothesis vs. Reality

**Initial Hypothesis:** Geography (Country Frequency) would be a primary fraud indicator.

**Model Reality:** Country frequency did **not** make the top 5 drivers.

### Why the Model "Ignored" Geography

#### 1. Granularity vs. Noise

- **Country Frequency:** High-bias feature — treating entire nations as risk units creates too many false positives
- **Device Frequency:** High-precision feature — identifies specific entities with suspicious behavior
- **Result:** Model chooses "surgical strikes" over "blanket judgments"

#### 2. Feature Dominance (Redundancy)

- High-risk actors from any country often use the same device to cycle through stolen credentials
- Since `device_id_freq` captures this pattern with ~51% importance, country adds marginal value
- Model prioritizes the "smoking gun" (device) over "background noise" (country)

#### 3. Adversarial Adaptation

- Modern fraudsters frequently use VPNs and residential proxies to mask location
- IP-based geography is easily faked and therefore unreliable
- **Time Since Signup** and **Device Fingerprinting** are much harder to manipulate

### Key Takeaway

> **Our model is built on intent and behavior rather than demographics.** This makes the system more robust against VPN usage and reduces the risk of geographical bias, ensuring we don't accidentally alienate entire legitimate markets based on location alone.

---

## 📈 6. Business Rules Translation

### From SHAP Insights to Actionable Rules

SHAP analysis enables translation of model behavior into simple business logic:

**Example Rule 1:**

> Flag all accounts where a purchase occurs within **10 minutes of creation** from a device that has been used by **5+ different accounts** in the past 24 hours.

**Example Rule 2:**

> High-risk transaction if `device_id_freq > threshold` AND `time_since_signup < 1 hour`.

### Benefits

- **Regulatory Compliance:** Clear, auditable rules for stakeholders
- **Model Monitoring:** Rules can be implemented as real-time checks
- **Human-in-the-Loop:** Security teams can override based on context

---

## 🎯 7. Key Findings Summary

### Behavioral Patterns Dominate

- **Device reuse** (`device_id_freq`) is the strongest fraud signal (~51% importance)
- **Account age** (`time_since_signup`) is the second strongest (~37% importance)
- Together, these behavioral features account for **~88% of predictive power**

### Geography is Less Predictive Than Expected

- Country frequency did not make the top 5 drivers
- Model learned that behavioral signals are more reliable than geographic ones
- Reduces risk of geographical bias and false positives

### Model Learns Realistic Fraud Behaviors

- High device reuse signals coordinated fraud rings or botnets
- Short time since signup indicates "hit-and-run" fraud tactics
- Other features provide contextual support but are not decisive alone

### Explainability Enables Trust

- SHAP provides transparent, interpretable explanations
- Stakeholders can verify model reasoning
- Facilitates regulatory compliance and audit requirements

---

## 🏆 8. Business Value Delivered

- ✔ **Transparent AI:** SHAP explanations build trust with stakeholders
- ✔ **Actionable Insights:** Feature importance guides business rule creation
- ✔ **Reduced Bias:** Model prioritizes behavior over demographics
- ✔ **Operational Efficiency:** Understanding FP/FN cases improves model refinement
- ✔ **Regulatory Compliance:** Explainable AI meets audit and governance requirements
