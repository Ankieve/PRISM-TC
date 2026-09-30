<div align="center">

# 🌀 PRISM-TC · VAYUVEGA

### AI Cyclone Intelligence: satellite-image intensity classification with an honest, transparent forecast dashboard

*Smart India Hackathon 2026 · Problem Statement **SIH26070** · North Indian Ocean basin*

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Model](https://img.shields.io/badge/Model-EfficientNet--B0-EE4C2C?logo=pytorch&logoColor=white)
![Map](https://img.shields.io/badge/Map-Leaflet-199900?logo=leaflet&logoColor=white)
![Status](https://img.shields.io/badge/Status-Hackathon_MVP-ff69b4)
![Transparency](https://img.shields.io/badge/Every_value-tagged_MODEL_or_SIMULATED-brightgreen)

[🚀 Live Demo](https://prism-tc.onrender.com) · [🧪 Run Locally](#-run-it-locally) · [🔍 Real vs Simulated](#-whats-real-and-whats-simulated) · [👥 Team](#-team-vayuvega)

</div>

---

## ✨ What is this?

**PRISM-TC** (dashboard name: **VAYUVEGA**) looks at a **satellite image of a tropical cyclone**, runs it through a deep-learning model, and shows the result on an interactive dashboard: intensity class, confidence, class probabilities, a past track and a 24-hour outlook.

> 💡 **Our rule: no fake confidence.** Every value on screen carries a tag, **MODEL** (from the AI), **SIMULATED** (historical-analog / rule-based) or **EST.** (estimated from the predicted class), so nobody is ever misled about what the AI actually predicted.

---

## 🎯 Features

| | Feature | What it does |
|---|---|---|
| 🛰️ | **AI intensity classification** | EfficientNet-B0 wind-regression ensemble predicts the cyclone intensity class from an image |
| 📊 | **Confidence + class probabilities** | Shows how sure the model is, not just its top answer (with a note that confidence can run high) |
| 🗺️ | **Live track map** | Observed IBTrACS track, simulated 24 h outlook and current position on a Leaflet map |
| 📈 | **Intensity outlook** | Simulated max sustained wind for the next 24 hours |
| ⚡ | **Rapid-intensification check** | Rule-based environment check (SST, wind shear, humidity) with 850 hPa vorticity shown for context only |
| 🎬 | **Judge Demo Mode** | 3 prepared scenarios that run without needing internet or a good example image on hand |
| 🚫 | **OOD rejection** | Invalid, non-cyclone documents are rejected instead of being force-classified |
| 🧾 | **Data provenance** | See the image source, storm ID and position behind every prediction |
| 🧪 | **Backtest & Transparency pages** | Model performance computed from the project's own files, nothing filled in by hand |
| 🛰️ | **Satellite view** | Live INSAT visualization |

### 🎬 Judge Demo scenarios
1. **Valid TCIR sample → AI classification**
2. **Invalid document → OOD rejection**
3. **Live satellite → INSAT visualization**

---

## 🔍 What's real and what's simulated

| Output | Source | Tag |
|---|---|---|
| Intensity class | 🤖 EfficientNet-B0 ensemble | `MODEL` |
| Confidence | 🤖 EfficientNet-B0 ensemble | `MODEL` |
| Class probabilities | 🤖 EfficientNet-B0 ensemble | `MODEL` |
| Wind | 📏 Class range from the predicted category | `EST.` |
| Track outlook | 📚 Copied from a similar historical storm (IBTrACS) | `SIMULATED` |
| Intensity (wind) outlook | 📚 Historical-analog | `SIMULATED` |
| Pressure | 📏 Rule-based | `SIMULATED` |
| Risk index | 📏 Rule-based | `SIMULATED` |
| Rapid-intensification flag | 📏 Rule-based | `SIMULATED` |

### 📈 Model performance
- **53.7%** exact-class match
- **93.1%** within one class
- Measured on a **12-storm, 520-image, storm-wise held-out test set** (the model never saw these storms in training)

### 🧮 Dataset at a glance
- **3,205** storm observations (every 3 h) · **75** distinct storms · North Indian Ocean, seasons **2003-2016**
- Max wind range **15-145 kt** (mean 40.9 kt)
- **8.5×** class imbalance (Depression vs Extremely Severe). Always guessing *Depression* would score **48%**, which is why we also report within-one-class accuracy.

### 🌊 Data in use
**TCIR** (train/test) · **IBTrACS** (labels) · **MOSDAC / INSAT** (Biparjoy validation + live view) · **ERA5** (environment check)

---

## 🗂️ Project structure

```
PRISM-TC-Frontend/
├── index.html · style.css · app.js     # 🎨 the dashboard
├── backend/
│   ├── server.py                       # 🌐 serves the dashboard AND the /api routes
│   ├── logic.py                        # 🧭 track / risk / class logic
│   ├── predict.py                      # 🤖 the AI model call (needs model.pth, NO normalization)
│   ├── model.pth                       # ⚠️ copy this in (final model from Drive → scripts/)
│   ├── data/                           # 🌪️ historical storm positions (IBTrACS)
│   ├── samples/                        # 🖼️ test images for the demo
│   ├── tests/smoke_test.py             # ✅ sanity checks
│   └── requirements.txt
├── CLAUDE.md                           # 📝 notes for Claude (read first if you use it)
└── _original_frontend_backup/          # 💾 original design files, untouched
```

---

## 🧪 Run it locally

> Steps written for **Windows**, run from the project folder.

**1️⃣ Install dependencies**
```bash
pip install -r backend/requirements.txt
```

**2️⃣ Add the model** 🧠
Copy `model.pth` into the `backend/` folder.

**3️⃣ Start the server**
```bash
python backend/server.py
```

**4️⃣ Open the dashboard** 🌐
Go to **http://localhost:8000**, then either:
- ▶️ press **Run Demo**, or
- 🖼️ pick a sample image / upload your own and press **Generate AI Prediction**

### 🚦 Status indicator
The top-left engine card and the sidebar show **ONLINE · model loaded** when the real model is running. Without `model.pth` the app runs in **DEMO MODE** with clearly labelled placeholder predictions.

### ✅ Smoke tests
```bash
python backend/tests/smoke_test.py          # check the backend
python backend/tests/smoke_test.py --real   # check with the real model
```

---

## 🌍 Live deployment

Deployed on Render: **https://prism-tc.onrender.com**

> ⏳ Free-tier instances can take a little while to wake up on the first visit.

### 🌐 Internet & offline behaviour
Map and chart libraries and fonts load from CDNs. With no internet, the page **falls back** to a simple track plot and chart, so the demo keeps working. 🙌

---

## 🧰 Tech stack

- 🎨 **Frontend:** HTML, CSS, JavaScript, Leaflet + OpenStreetMap
- ⚙️ **Backend:** Python web server with `/api` routes
- 🤖 **Model:** EfficientNet-B0 (PyTorch, `model.pth`)
- 🌪️ **Data:** TCIR, IBTrACS, MOSDAC/INSAT, ERA5

---

## 👥 Team VAYUVEGA

| | Member |
|---|---|
| 🌀 | **Binayak Mandal** |
| 🌀 | **Kaushikee Karmakar** |
| 🌀 | **Ankita Kundu** |
| 🌀 | **Anurag Tiwari** |
| 🌀 | **Loknath Acharya** |
| 🌀 | **Kripa Das** |

Built with 💙 for **Smart India Hackathon 2026** (SIH26070).

<!-- Add a screenshot here:
![Dashboard screenshot](docs/dashboard.png)
-->

<div align="center">

🌀 *Turning satellite pixels into cyclone insight, honestly.* 🌀

</div>
