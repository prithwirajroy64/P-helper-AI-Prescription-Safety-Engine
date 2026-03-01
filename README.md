# P-helper — AI-Powered Prescription Safety Engine

## What's used: 5 ML/NLP Modules

| # | Module | Algorithm | Function |
|---|--------|-----------|----------|
| 1 | **Medical NER** | Logistic Regression, BIO-tagger, char n-gram TF-IDF | Extracts DRUG, DOSE, FREQ, ROUTE, DURATION, CONDITION, ALLERGY |
| 2 | **Drug Name Normalizer** | TF-IDF cosine similarity (char bigrams/trigrams) | Handles typos, Indian brands, abbreviations → generic |
| 3 | **DDI Severity Classifier** | Logistic Regression, 9 pharmacological features | MAJOR/MODERATE/MINOR + confidence + explanation |
| 4 | **Polypharmacy Risk Scorer** | Gradient Boosting, 12 features | LOW/MOD/HIGH/CRITICAL + 0-100 score + SHAP factors |
| 5 | **Clinical Sentence Classifier** | Complement Naive Bayes, TF-IDF bigrams | Labels sentences: DRUG_NAME / DOSAGE / ALLERGY / DIAGNOSIS |

All models train in **< 0.5 seconds** at startup. No external model files needed.

## Quick Start

```bash
# 1. Install
pip install flask scikit-learn numpy pdfplumber

# 2. Run
python app.py

# 3. Open browser
# http://localhost:5000
```

## Project Structure

```
p_helper/
├── app.py              ← Flask app with full ML/NLP integration
├── nlp_engine.py       ← 5 ML/NLP modules (self-contained)
├── brand_names.py      ← Brand→generic resolution (Indian + global)
├── requirements.txt
└── static/
    ├── css/style.css
    └── js/main.js
templates/
└── index.html
```

## How ML Integrates with the Original Code

The original `app.py` pipeline (API calls, regex parser, `detect_errors`) is **fully preserved**.

The ML layer adds a new `/analyze` step at the end:

```python
# After API-based detection:
ml_result = pipeline.run(text, enriched_drugs, api_alerts, drug_info_map)

# ML-detected interactions are MERGED into alerts (no duplicates):
for ml_int in ml_result["ml_interactions"]:
    if pair not in existing_api_pairs:
        alerts.append(ml_alert)
```

This means:
- API data (RxNorm, OpenFDA, RxClass) = ground truth
- ML = additional signal that catches what APIs miss
- No conflicts, fully additive

## API Sources (Original, Preserved)

| API | URL | Used For |
|-----|-----|---------|
| RxNorm | `rxnav.nlm.nih.gov/REST/rxcui.json` | Drug normalization + RxCUI |
| RxClass | `rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json` | Pharmacological classes |
| OpenFDA Label | `api.fda.gov/drug/label.json` | Dosage text, warnings, interactions |
| OpenFDA FAERS | `api.fda.gov/drug/event.json` | Adverse event signals |

Network auto-detection: online → live APIs; offline → mock server.

## Download Reports

- **PDF** — Full printable HTML clinical report (open in browser → print/save PDF)
- **CSV** — Drugs, alerts, ML interactions, NER entities (Excel-compatible)
- **JSON** — Complete raw ML+API output for EHR integration

## Disclaimer
Clinical decision support only. Not a substitute for professional pharmacist/physician judgment.
