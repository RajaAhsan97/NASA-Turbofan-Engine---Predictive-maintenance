# Predictive Maintenance for NASA Turbofan Engines — Webinar

## What's included
- `predictive_maintenance_nasa_turbofan.ipynb` — the full end-to-end notebook (sections 1–6 + dashboard hand-off)
- `requirements.txt` — all dependencies

## 1. Get the data
Download the NASA C-MAPSS Turbofan dataset (FD001 used by default) from either:
- Kaggle: "NASA Turbofan Jet Engine Data Set"
- NASA Prognostics Data Repository

To demo a different subset (more complex, multi-regime), change `DATASET_ID` in the notebook's Section 0 to `"FD002"`, `"FD003"`, or `"FD004"` and supply the matching files.

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Run the notebook (Sections 0–7)
Open `predictive_maintenance_nasa_turbofan.ipynb` and run all cells top to bottom. This will:
- Explore and clean the data
- Train a baseline and several advanced RUL models
- Save the best model to `models/best_rul_model.pkl` (+ scaler and feature metadata)
- Train sensor forecasters and project future RUL at +2 / +5 cycles
- Demo the GenAI maintenance assistant (works with a rule-based fallback if no Gemini key is set)

## 4. Enable the GenAI assistant (optional but recommended for the webinar)
Get a free Gemini API key: https://aistudio.google.com/app/apikey
```bash
export GOOGLE_API_KEY="your-key-here"
```
Without a key, both the notebook and the dashboard automatically fall back to a rule-based briefing so the demo never breaks live.

## 5. Launch the dashboard
```bash
streamlit run app.py
```
The dashboard reads the model artifacts saved by the notebook and the raw `data/` files, so run the notebook at least once first.

## Notes
- The RUL target uses the standard piecewise-linear clip (cap = 125 cycles) used throughout C-MAPSS literature — worth a 30-second explanation slide.
- The scoring includes the official NASA asymmetric scoring function alongside RMSE/MAE/R² — good talking point on why late predictions are riskier than early ones in maintenance.
- Sensor forecasting uses lag-feature Random Forests per sensor (fast, robust, easy to explain live); mention LSTM/Prophet as "next step" extensions if time allows.
- The GenAI assistant section is a good place to pause for audience Q&A — try asking it about a different engine live.
