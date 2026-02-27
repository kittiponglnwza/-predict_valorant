# Football AI — Nexus Engine v9.0

ระบบวิเคราะห์และทำนายผลฟุตบอล Premier League ด้วย Machine Learning  
รันด้วย Streamlit · Python 3.10+

---

## โครงสร้างโปรเจกต์

```
PROJECT AI/
│
├── data/                          # ชุดข้อมูลแมตช์ CSV รายฤดูกาล
│   ├── season 2020.csv
│   ├── season 2021.csv
│   ├── ...
│   └── season 2025.csv
│
├── doc/                           # เอกสารประกอบ
│   ├── doc.txt
│   ├── how to use git.txt
│   └── test version 1
│
├── model/                         # โฟลเดอร์เก็บ model ที่ train แล้ว (.pkl)
│
├── models/                        # โฟลเดอร์เก็บ model สำรอง / เวอร์ชันต่างๆ
│
├── src/                           # Business Logic หลักทั้งหมด
│   ├── __init__.py
│   ├── config.py                  # ค่าคงที่, API Key, path ต่างๆ
│   ├── features.py                # Feature Engineering pipeline
│   ├── model.py                   # Train / Load model
│   ├── predict.py                 # ฟังก์ชันทำนาย, simulation, API calls
│   └── analysis.py                # Monte Carlo, ROI backtest, calibration
│
├── ui/                            # Streamlit UI (แยกเป็นรายหน้า)
│   ├── __init__.py
│   ├── app_ui.py                  # Entry point — รันไฟล์นี้
│   ├── styles.py                  # Global CSS
│   ├── sidebar.py                 # Sidebar navigation & system status
│   ├── utils.py                   # Helper functions ที่ใช้ร่วมกัน
│   ├── page_overview.py           # หน้า Dashboard
│   ├── page_predict.py            # หน้า Match Prediction
│   ├── page_fixtures.py           # หน้า Upcoming Fixtures
│   ├── page_season.py             # หน้า Season Table + AI Simulation
│   ├── page_analysis.py           # หน้า Model Analysis & Insights
│   └── page_update.py             # หน้า Data Management
│
├── main.py                        # Script รัน pipeline แบบ CLI
├── test_fixtures.py               # Unit test สำหรับ fixtures
├── test_standings.py              # Unit test สำหรับ standings
└── README.md                      # ไฟล์นี้
```

---

## UI Structure (`ui/`)

โฟลเดอร์ `ui/` ถูกแยกออกจาก `app_ui.py` ไฟล์เดียว (1,083 บรรทัด) ให้เป็นไฟล์ย่อยตามหน้าที่รับผิดชอบ เพื่อให้พัฒนาและดูแลรักษาได้ง่ายขึ้น

### `app_ui.py` — Entry Point
ไฟล์หลักที่ใช้รัน Streamlit ทำหน้าที่เพียง 3 อย่าง:
1. ตั้งค่า `st.set_page_config` และ inject Global CSS
2. โหลด / Train model ผ่าน `@st.cache_resource`
3. Route ไปยังหน้าที่เลือกผ่าน `session_state['nav_page']`

```bash
streamlit run ui/app_ui.py
```

---

### `styles.py` — Global CSS
รวม CSS ทั้งหมดที่ใช้ร่วมกันทุกหน้า เช่น สีพื้นหลัง, metric card, tab, button, sidebar  
เรียกใช้ครั้งเดียวตอนเริ่ม app:

```python
from ui.styles import inject_global_css
inject_global_css()
```

---

### `sidebar.py` — Sidebar
แสดง Navigation radio และ System Status (Accuracy, Hybrid Mode, α, Features)  
เชื่อมกับ `st.session_state['nav_page']` เพื่อ routing:

```python
from ui.sidebar import render_sidebar
render_sidebar(ctx)
```

---

### `utils.py` — Helper Functions
ฟังก์ชัน utility ที่หลายหน้าใช้ร่วมกัน:

| ฟังก์ชัน | หน้าที่ |
|---|---|
| `silent(fn, *args)` | รัน function โดยซ่อน stdout output |
| `make_styled_table(df, pts_col, max_pts)` | สร้าง Pandas Styler สำหรับตารางลีก |
| `zone_label(pos)` | แปลง position → emoji zone label |
| `find_team_col(df)` | หา column ชื่อทีมใน DataFrame |

---

### หน้าต่างๆ

| ไฟล์ | หน้า | ฟังก์ชันหลัก |
|---|---|---|
| `page_overview.py` | Dashboard | Live matches, prediction stats, confusion matrix, top Elo |
| `page_predict.py` | Match Prediction | เลือกทีม → ดูความน่าจะเป็น, xG, สกอร์ที่คาด, form 5 นัด |
| `page_fixtures.py` | Upcoming Fixtures | ดึงตารางแข่งจาก API พร้อมปุ่ม Predict ส่งไปหน้า Predict |
| `page_season.py` | Season Table | ตารางคะแนนปัจจุบัน (API) + AI Simulation จนจบฤดูกาล |
| `page_analysis.py` | Analysis & Insights | Monte Carlo, ROI Backtest, Draw Calibration, Feature Importance |
| `page_update.py` | Data Management | Sync ข้อมูลจาก API, แสดงรายการ CSV ใน `/data` |

---

## การ Navigate ระหว่างหน้า

ใช้ `st.session_state` เป็น state กลาง:

```python
# เปลี่ยนหน้าจากโค้ด (เช่น ปุ่ม Predict ในหน้า Fixtures)
st.session_state['nav_page'] = "Predict Match"
st.session_state['pred_home'] = "Arsenal"
st.session_state['pred_away'] = "Chelsea"
st.session_state['auto_predict'] = True
```

---

## การเพิ่มหน้าใหม่

1. สร้างไฟล์ `ui/page_xxx.py` พร้อมฟังก์ชัน `def page_xxx(ctx):`
2. Import ใน `app_ui.py`
3. เพิ่มใน dict `pages` และ list ใน `render_sidebar()`

```python
# app_ui.py
from ui.page_xxx import page_xxx

pages = {
    ...
    "New Page": page_xxx,   # เพิ่มตรงนี้
}
```

```python
# sidebar.py — เพิ่มชื่อหน้าใน radio
st.radio("Navigation", [
    "Overview", "Predict Match", ..., "New Page"   # เพิ่มตรงนี้
], key="nav_page")
```

---

## Dependencies หลัก

```
streamlit
pandas
numpy
scikit-learn
requests
```

---

## การรัน

```bash
# ติดตั้ง dependencies
pip install -r requirements.txt

# รัน Streamlit app
streamlit run ui/app_ui.py
```


git add .
git commit -m " ดึง api  27/2/69 "
git push


้how to run  ui 

streamlit run app_ui.py

python -m streamlit run ui/app_ui.py


and program

python app.py

## STABILIZE (Accuracy-First)

```bash
# optional: force market features on/off
set USE_MARKET_FEATURES=1

# optional: tune target metric and recall guardrails
set STABILIZE_SELECTION_METRIC=accuracy
set STABILIZE_MIN_RECALL_DRAW=0.10
set STABILIZE_MIN_RECALL_HOME=0.08
set STABILIZE_MIN_RECALL_AWAY=0.08

# run rolling-origin backtest + tuning + profile selection
python -m pipelines.train_pipeline
```

Notes:
- Pipeline now compares `no_market` vs `with_market` profile and selects by average validation accuracy.
- Ensemble tuning includes LightGBM, CatBoost (if installed), and Poisson blend.
- Holdout uses frozen settings from validation only (draw weight, sigmoid flag, blend weights, thresholds).
