# 🌙 SleepSense AI — Sleep Disorder Prediction System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![HTML](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-yellow?style=flat-square&logo=html5)
![Model](https://img.shields.io/badge/Model-Deep%20Neural%20Network-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **An AI-powered web application that predicts sleep disorders using a trained Deep Neural Network — runs entirely in the browser with no backend required.**

---

## 📌 Project Overview

SleepSense AI ek complete end-to-end machine learning project hai jo kisi bhi insaan ke **lifestyle aur health data** ke basis par predict karta hai ki usse koi sleep disorder hai ya nahi.

Ek trained Keras model (`sleep_disorder_model.h5`) ke weights ko directly browser mein embed karke **pure JavaScript** se neural network inference run kiya jaata hai — koi server, koi API call nahi.

---

## 🎯 Problem Statement

Sleep disorders (insomnia, sleep apnea) ek growing health concern hain jinka early detection bahut zaroori hai. Traditional diagnosis ke liye expensive sleep studies (polysomnography) ki zaroorat hoti hai. Is project ka goal hai:

- Daily lifestyle data se **early-stage screening** karna
- Doctors ko **initial assessment tool** dena
- Patients ko **self-awareness** provide karna

---

## 🏷️ Output Classes

| Class | Disorder | Description |
|-------|----------|-------------|
| **0** | ✅ No Disorder | Sleep patterns normal range mein hain |
| **1** | 💤 Insomnia | Neend aane/rehne mein persistent difficulty |
| **2** | 😮 Sleep Apnea | Neend mein baar baar saans rukna |

---

## 🧠 Model Architecture

```
Input (49 features)
        │
   Dense(50, linear)       ← 2,500 params
        │
   Dense(128, ReLU)        ← 6,528 params
        │
   Dense(64, ReLU)         ← 8,256 params
        │
   Dense(32, ReLU)         ← 2,080 params
        │
   Dense(3, ReLU*)         ← 99 params
        │
   Output (3 classes)

Total Trainable Parameters: 19,463
```

> ⚠️ **Known Issue:** Output layer `ReLU` ki jagah `Softmax` honi chahiye thi — isse model Sleep Apnea ki taraf biased ho jaata hai. Fix ke liye [Retraining Guide](#-known-issues--improvements) dekho.

---

## 📊 Feature Engineering — 49 Input Features

Model ko **Sleep Health and Lifestyle Dataset** (Kaggle) par train kiya gaya hai. Features aur unki encoding:

| # | Feature | Type | Range / Encoding |
|---|---------|------|-----------------|
| 0 | Gender_Female | One-hot | 0 or 1 |
| 1 | Gender_Male | One-hot | 0 or 1 |
| 2 | Age | Normalized | (age − 18) / 62 |
| 3–13 | Occupation | One-hot (11 categories) | Accountant, Doctor, Engineer, Lawyer, Manager, Nurse, Sales Rep, Salesperson, Scientist, Software Engineer, Teacher |
| 14 | Sleep Duration | Normalized | (hrs − 4) / 6 |
| 15 | Quality of Sleep | Normalized | (score − 1) / 9 |
| 16 | Physical Activity Level | Normalized | min/day ÷ 120 |
| 17 | Stress Level | Normalized | (score − 1) / 9 |
| 18–21 | BMI Category | One-hot (4 categories) | Normal, Normal Weight, Obese, Overweight |
| 22 | Blood Pressure (Systolic) | Normalized | (mmHg − 80) / 120 |
| 23 | Blood Pressure (Diastolic) | Normalized | (mmHg − 50) / 90 |
| 24 | Heart Rate | Normalized | (bpm − 40) / 90 |
| 25 | Daily Steps | Normalized | (steps − 1000) / 19000 |
| 26–48 | Reserved / Padding | — | 0 |

---

## 🗂️ Project Structure

```
sleep-disorder-predictor/
│
├── 📄 sleep_disorder_model.h5      # Trained Keras model
├── 🌐 sleep_disorder_ui.html       # Complete frontend (self-contained)
├── 📓 README.md                    # This file
│
├── 📁 training/ (optional)
│   ├── train_model.py              # Model training script
│   ├── preprocess.py               # Feature engineering
│   └── sleep_health_data.csv       # Dataset (Kaggle)
│
└── 📁 assets/
    └── screenshots/
```

---

## 🚀 How to Run

### Option 1 — Direct Browser (No Setup Required)
```
1. sleep_disorder_ui.html file download karo
2. Browser mein open karo (double-click)
3. Patient data fill karo
4. "Analyze Sleep Health" click karo
```
✅ Koi server, koi Python, koi installation — kuch bhi nahi chahiye.

---

### Option 2 — Python Model (Development)

**Requirements install karo:**
```bash
pip install tensorflow numpy scikit-learn pandas
```

**Model load karke predict karo:**
```python
import tensorflow as tf
import numpy as np

# Model load karo
model = tf.keras.models.load_model('sleep_disorder_model.h5')

# Sample input (49 features — normalized)
# [female, male, age, *occupations(11), sleep_dur, sleep_qual,
#  activity, stress, *bmi(4), bp_sys, bp_dia, hr, steps, *zeros(23)]
sample = np.zeros(49)
sample[1] = 1       # Male
sample[2] = 0.27    # Age ~35
sample[4] = 1       # Engineer
sample[14] = 0.5    # Sleep 7h
sample[15] = 0.67   # Quality 7/10
sample[16] = 0.38   # Activity 45 min
sample[17] = 0.44   # Stress 5/10
sample[21] = 1      # Overweight
sample[22] = 0.37   # BP 124/80
sample[23] = 0.33   # BP diastolic
sample[24] = 0.36   # HR 72
sample[25] = 0.32   # 7000 steps

prediction = model.predict(sample.reshape(1, -1))
classes = ['No Disorder', 'Insomnia', 'Sleep Apnea']
result = classes[np.argmax(prediction)]
print(f"Predicted: {result}")
```

---

## 🌐 Frontend — How It Works

`sleep_disorder_ui.html` ek **fully self-contained** file hai jisme:

1. **Model weights** (19,465 parameters) JSON format mein embedded hain
2. **Pure JavaScript neural network** jo exact same forward pass karta hai jaise TensorFlow
3. **Temperature-scaled softmax** confidence display ke liye
4. **Zero dependencies** — koi CDN, koi library import nahi

### JS Inference Pipeline:
```
User Input
    ↓
Feature Encoding (one-hot + normalization)
    ↓
Matrix Multiplication (transposed kernels)
    ↓
ReLU Activations
    ↓
Temperature Softmax (T=0.05)
    ↓
Probability Display + Diagnosis
```

---

## ⚠️ Known Issues & Improvements

### 🔴 Issue 1: Model Sleep Apnea ki taraf biased hai

**Reason:** Output layer mein `ReLU` hai `Softmax` ki jagah. Class 2 ka raw output average `0.27` hai vs Class 1 ka sirf `0.008`.

**Fix — Model ko sahi se retrain karo:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils.class_weight import compute_class_weight

model = Sequential([
    Dense(50, activation='relu', input_shape=(49,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')   # ← ReLU ki jagah SOFTMAX
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Class imbalance fix
weights = compute_class_weight('balanced',
                                classes=[0, 1, 2],
                                y=y_train)
class_weight = dict(enumerate(weights))

model.fit(X_train, y_train,
          epochs=50,
          batch_size=32,
          class_weight=class_weight,    # ← Important!
          validation_split=0.2)
```

### 🟡 Issue 2: 23 features zeros hain

Features 26–48 hamesha 0 hain kyunki original training dataset ka exact encoding unknown hai. Model accuracy badhegi agar full feature set recover ho.

### 🟢 Improvement Ideas

| Improvement | Description |
|-------------|-------------|
| SMOTE oversampling | Minority class (Insomnia) ke synthetic samples banana |
| Batch Normalization | Training stability improve karna |
| Dropout layers | Overfitting rokna |
| Cross-validation | Model reliability test karna |
| SHAP values | Feature importance explain karna |

---

## 📈 Dataset Info

| Property | Value |
|----------|-------|
| **Source** | Sleep Health and Lifestyle Dataset (Kaggle) |
| **Samples** | ~400 records |
| **Features** | 13 raw → 49 after encoding |
| **Target** | Sleep Disorder (3 classes) |
| **Class Distribution** | No Disorder ~58%, Insomnia ~21%, Sleep Apnea ~21% |

---

## 🖥️ UI Features

- 🎚️ **Interactive sliders** — Age, Sleep Duration, Quality, Activity, Stress, Steps
- 🔘 **Gender toggle** — Male / Female
- 📋 **Dropdowns** — Occupation (11 options), BMI Category (4 options)
- 🔢 **Number inputs** — Systolic BP, Diastolic BP, Heart Rate
- 📊 **Animated probability bars** — Teen classes ke liye real-time visualization
- 💊 **Condition info panel** — Tabs mein har disorder ki symptoms
- 🏗️ **Model stats card** — Architecture details
- 🌙 **Dark theme** — Professional medical UI

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | TensorFlow / Keras |
| Model Format | HDF5 (`.h5`) |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Inference | Pure JS (no TF.js needed) |
| Fonts | Google Fonts (DM Serif Display, Syne, IBM Plex Mono) |
| Deployment | Static file — no server needed |

---

## 📋 Medical Disclaimer

> ⚠️ **Ye tool sirf educational aur research purposes ke liye hai।**
> Yahan diye gaye predictions kisi bhi rup mein professional medical advice, diagnosis, ya treatment ka substitute nahi hain। Agar aapko sleep-related symptoms hain toh kripya ek qualified healthcare professional se sampark karein।

---

## 👨‍💻 Author

Developed as part of an ML deployment project demonstrating:
- Keras model training & export
- Model weight extraction & browser embedding
- Pure JS neural network inference
- Production-grade frontend UI design

---

## 📄 License

MIT License — Free to use, modify, and distribute with attribution.
