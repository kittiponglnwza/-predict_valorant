# Football AI — Nexus Engine v9.0

ระบบวิเคราะห์และทำนายผลฟุตบอล Premier League ด้วย Machine Learning + Neural Networks
รันด้วย Streamlit · Python 3.10+

---

# 🗂 Dataset (ชุดข้อมูลและตัวแปรที่ใช้)

ระบบใช้ข้อมูลการแข่งขันย้อนหลังของ Premier League ควบคู่กับการดึงข้อมูลแบบ Real-time ผ่าน API เพื่อให้โมเดลมีข้อมูลที่ครอบคลุมทั้งสถิติในอดีตและปัจจุบัน 

## 1. แหล่งข้อมูลหลัก (Data Sources)
- **Historical Data (CSV):** ข้อมูลย้อนหลังเก็บในโฟลเดอร์ `data/` (เช่น `season 2020.csv` ถึง `season 2025.csv`) รวมข้อมูลฝึกสอนมากกว่า ~2,000 แมตช์
- **Live API Integration:**
  - `update_season_csv_from_api`: ดึงผลการแข่งขันและอัปเดตไฟล์ CSV ของฤดูกาลปัจจุบันอัตโนมัติ
  - `get_pl_standings_from_api`: ดึงตารางคะแนนล่าสุดแบบ Real-time
  - **Crests CDN:** ดึงรูปภาพตราสโมสรจาก `football-data.org` เพื่อใช้แสดงผลบน UI

## 2. ข้อมูลพื้นฐานของการแข่งขัน (Raw Match Data)
- **ข้อมูลแมตช์:** ไอดีแมตช์ (`MatchID`), วันที่แข่ง (`Date`), ทีมเหย้า (`HomeTeam`), ทีมเยือน (`AwayTeam`)
- **ผลการแข่งขัน:** ประตูฝั่งเจ้าบ้าน (`FTHG`), ประตูฝั่งทีมเยือน (`FTAG`), และผลลัพธ์ของเกม (`FTR`: H, D, A)

## 3. ข้อมูลสถิติขั้นสูงและราคาต่อรอง (Advanced Stats & Market Odds)
- **Expected Goals (xG):** ความน่าจะเป็นของการได้ประตูของแต่ละทีม (`HomeXG`, `AwayXG`)
- **Betting Odds (ราคาพูล 1x2):** อัตราต่อรองจากสำนักต่างๆ เช่น Bet365, Pinnacle, William Hill และค่าเฉลี่ยตลาด (`AvgH`, `AvgD`, `AvgA`)
- **Market Features:** - `Implied Probability`: ความน่าจะเป็นแฝงที่คำนวณจากราคาต่อรอง (`_ImpH`, `_ImpD`, `_ImpA`)
  - `Overround`: อัตราความได้เปรียบ (Margin) ของเจ้ามือ

## 4. ตัวแปรเชิงสถิติที่ระบบสร้างขึ้น (Engineered Features)
ระบบสร้างฟีเจอร์ขั้นสูงกว่า 100+ ตัวแปรแบบ Time-safe (ไม่มี Data Leakage) เพื่อป้อนให้ Machine Learning:
* **Elo Ratings:** คะแนนความเก่งแบบไดนามิก อัปเดตหลังจบทุกแมตช์ แยกเป็นภาพรวม (`Elo_HA`), ฟอร์มในบ้าน (`Home_Elo_H`), ฟอร์มนอกบ้าน (`Away_Elo_A`), และส่วนต่างคะแนน (`Elo_Diff`)
* **Rolling Form (ฟอร์มย้อนหลัง 5-10 นัด):** - สถิติพื้นฐาน: ประตูที่ทำได้ (`GF5`), ประตูที่เสีย (`GA5`), แต้มสะสม (`Pts5`, `Pts10`)
  - สถิติเชิงลึก: อัตราชนะ (`Win5`), คลีนชีต (`CS5`), การทำประตูต่อเนื่อง (`Scored5`), ยิงเข้ากรอบ (`HST5`)
* **Exponential Moving Average (EWM):** การถ่วงน้ำหนักสถิติโดยให้ความสำคัญกับแมตช์ล่าสุดมากกว่าแมตช์เก่า (`GF_ewm`, `Pts_ewm`)
* **xG Form (ฟอร์ม xG เชิงลึก):** ค่าเฉลี่ย xG 5 นัดหลัง (`xGF5`, `xGA5`), แนวโน้มของ xG (`xGF_slope`), และความสามารถในการทำประตูจริงเทียบกับโอกาส (`xG_overperf`)
* **Head-to-Head (H2H):** อัตราชนะ/เสมอ เมื่อทั้งสองทีมนี้เจอกัน (`H2H_HomeWinRate`, `H2H_DrawRate`)
* **Contextual & Environment:** จำนวนวันพักก่อนแข่ง (`DaysRest`), โมเมนตัมของทีม (`Momentum`), และความได้เปรียบในการเป็นเจ้าบ้าน (`HomeAdvantage`)

## 5. การจัดการข้อมูลและการลดความเอนเอียง (Data Processing Pipeline)
- **Data Splitting:** แบ่ง Train/Test ชุดข้อมูลตามฤดูกาล (Season-based split) เพื่อทดสอบแบบ Forward-time เสมือนการทำนายในโลกความเป็นจริง
- **Class Imbalance Handling:** ใช้เทคนิค **SMOTE** เพื่อสังเคราะห์ข้อมูลแมตช์ "เสมอ (Draw)" เพิ่มเติม เนื่องจากผลเสมอมักจะเกิดขึ้นน้อยกว่าในชุดข้อมูลจริง ช่วยเพิ่มความแม่นยำ (Recall) ให้กับโมเดล
- **Clean Pipeline:** กรองแมตช์ที่ยังไม่เตะออกก่อนเทรน (Drop NaN ในคอลัมน์ FTR) และมีระบบ Bootstrap คะแนน Elo สำหรับทีมที่เพิ่งเลื่อนชั้น
---

## โครงสร้างข้อมูลดิบ

ตัวอย่างคอลัมน์หลัก:

* HomeTeam / AwayTeam
* FTHG / FTAG (Full Time Goals)
* Shots / Shots on Target
* Possession
* Match Date

---

## Feature Engineering (src/features.py)

ระบบสร้างฟีเจอร์ขั้นสูง เช่น:

* Rolling goals (3–5 นัดล่าสุด)
* Rolling points average
* Home / Away strength split
* Goal difference momentum
* Form-based signals
* Expected goal proxy (xG-like features)

ทุกฟีเจอร์ถูกสร้างแบบ time-safe
ไม่มีข้อมูลอนาคต (No Data Leakage)

---

# 🤖 Model Architecture

ระบบใช้ Hybrid / Ensemble Architecture

---

## 1️⃣ Core Machine Learning Engine

โมเดลหลัก:

* LightGBM / Gradient Boosting

### Two-Stage Model Structure

Stage 1 → ทำนาย: Draw vs Not Draw
Stage 2 → ถ้า Not Draw → ทำนาย: Home Win vs Away Win

เหตุผล:

* ผล "เสมอ" เป็นคลาสที่ทายยาก
* การแยก stage ช่วยเพิ่ม Recall และเสถียรภาพของโมเดล

---

## 2️⃣ Goal Model (Poisson Regression)

ใช้ Poisson Regressor เพื่อคำนวณ:

* λ_home (Expected Home Goals)
* λ_away (Expected Away Goals)

จากนั้นสร้าง:

* Scoreline Probability Matrix
* Over/Under Probability
* Win probability จากการรวมความน่าจะเป็นของสกอร์

---

## 3️⃣ Neural Network Boost (MLPClassifier)

ใช้ Multi-Layer Perceptron 2 เวอร์ชัน:

* Hidden Layers (128, 64)
* Hidden Layers (256, 128, 64)

NN ไม่ได้ใช้เดี่ยว ๆ
แต่ใช้ "Blending" รวมความน่าจะเป็นกับ LightGBM

ข้อดี:

* จับ nonlinear pattern ลึก ๆ
* เพิ่มความเสถียรของ F1-score
* ลด overfitting บางกรณี

---

## 4️⃣ Class Imbalance Handling

ใช้ SMOTE เพื่อแก้ปัญหา Draw มีจำนวนน้อย

---

## 5️⃣ Probability Calibration

ใช้ CalibratedClassifierCV (Isotonic)

เพื่อให้เปอร์เซ็นต์ความน่าจะเป็นมีความสมจริงมากขึ้น

---

# 📊 Evaluation Strategy

* Rolling-Origin Backtest (2020 → 2025)
* Accuracy
* Macro F1-Score
* Confusion Matrix
* ROI Simulation
* Monte Carlo Season Simulation

ประเมินแบบ forward-time เท่านั้น

---

# 🧠 System Identity

FOOTBALL AI v9.0 คือ Football Analytics Engine ที่รวม:

* Advanced Feature Engineering
* Hybrid ML + Neural Network
* Statistical Goal Modeling
* Time-Aware Validation
* Simulation & Backtesting

ระดับความเข้มข้น: Academic / Junior Data Scientist Level

---

# 🚀 Run

ติดตั้ง dependencies:

```
pip install -r requirements.txt
```

รัน Streamlit:

```
python -m streamlit run ui/app_ui.py
```

เทรน pipeline:

```
python -m pipelines.train_pipeline
```
