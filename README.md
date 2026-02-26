# ⚽ Football AI v9.0 — Premier League Predictor

## โครงสร้างโปรเจกต์

```
project_ai/
│
├── data/                    ← ไฟล์ CSV ทุก season
│   ├── season 2020.csv
│   ├── season 2021.csv
│   ├── season 2022.csv
│   ├── season 2023.csv
│   ├── season 2024.csv
│   └── season 2025.csv
│
├── models/                  ← saved model
│   └── football_model_v9.pkl
│
├── src/                     ← source code
│   ├── __init__.py
│   ├── config.py            ← imports, constants, paths, team map
│   ├── features.py          ← data loading + feature engineering
│   ├── model.py             ← training, calibration, save/load
│   ├── predict.py           ← predict_match, season sim, API
│   └── analysis.py          ← backtest, CV, monte carlo, SHAP
│
├── ui/                      ← Streamlit UI
│   ├── __init__.py
│   └── app_ui.py
│
├── main.py                  ← CLI entry point
└── README.md
```

## วิธีรัน

### CLI (Terminal)
```bash
cd project_ai
python main.py
```

### Streamlit UI
```bash
cd project_ai
streamlit run ui/app_ui.py
```

## ย้ายไฟล์จากโครงสร้างเก่า

| เก่า | ใหม่ |
|------|------|
| `data_set/*.csv` | `data/*.csv` |
| `model/*.pkl` | `models/*.pkl` |
| `config.py` | `src/config.py` |
| `features.py` | `src/features.py` |
| `model.py` | `src/model.py` |
| `predict.py` | `src/predict.py` |
| `analysis.py` | `src/analysis.py` |
| `app.py` | `main.py` |
| `app_ui.py` | `ui/app_ui.py` |

## Architecture

```
main.py
  │
  ├── src.features.run_feature_pipeline()
  │     └── load CSVs → ELO → rolling stats → xG features → match_df
  │
  ├── src.model.run_training_pipeline()
  │     └── Optuna tune → 2-Stage LightGBM → Poisson Hybrid → save pkl
  │
  ├── src.predict.run_season_simulation(ctx)
  │     └── predict remaining fixtures → projected table
  │
  ├── src.predict.predict_with_api(team, ctx)
  │     └── fetch fixtures API → predict next 5 matches
  │
  └── src.predict.show_next_pl_fixtures(ctx)
        └── fetch PL schedule → predict all upcoming
```

## Model

- **2-Stage LightGBM**: Stage1 = Draw vs Not-Draw, Stage2 = Home vs Away  
- **Poisson Hybrid**: blend ML + Poisson goals model (dynamic α)  
- **Adaptive Draw Suppression**: factor ปรับตาม bias จริง  
- **Threshold Optimization**: Optuna grid search macro F1  
- **Accuracy**: ~51% (vs baseline 33%)
