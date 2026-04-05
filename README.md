# ScamShield — Full Training Guide

Train a real scam detection model from scratch using the UCI SMS Spam Collection (public dataset, CC BY 4.0).

---

## Project Structure

```
scamshield/
├── train.py                      # Main English training script
├── train_multilingual.py         # Multilingual training (EN + HI/MR/TE/KN)
├── predict.py                    # Inference — load models and classify messages
├── generate_multilingual_data.py # Generate synthetic multilingual training data
├── api.py                        # Flask REST API server
├── requirements.txt
└── models/                       # Created after training
    ├── scam_detector_gbm32.pkl
    ├── scam_detector_ngram.pkl
    └── training_results.json
```

---

## Step 0 — Install Dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

---

## Step 1 — Download the Dataset

**UCI SMS Spam Collection** — 5,574 real SMS messages, public domain (CC BY 4.0).

### Option A — Direct download
```bash
# macOS / Linux
curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip" -o sms.zip
unzip sms.zip
# The file you need is: SMSSpamCollection  (tab-separated, no header)
```

### Option B — Kaggle
1. Go to https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Download `spam.csv`
3. Convert with this one-liner:
```python
import pandas as pd
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1","v2"]]
df.columns = ["label","message"]
df.to_csv("SMSSpamCollection", sep="\t", index=False, header=False)
```

### Option C — Manual
Create a file called `SMSSpamCollection` with tab-separated rows:
```
ham	Go until jurong point, crazy.. Available only in bugis n great world...
spam	WINNER!! As a valued network customer you have been selected...
```

### Verify your file
```bash
head -3 SMSSpamCollection
# Should show:  ham<TAB>message...  OR  spam<TAB>message...
wc -l SMSSpamCollection
# Should be ~5574
```

---

## Step 2 — Train the English Model

```bash
python train.py --data SMSSpamCollection --out models
```

**What this does:**
- Loads 5,574 real SMS messages
- Splits 80/20 (train=4,459, test=1,115)
- Extracts 24 handcrafted features per message
- Trains a char 3-5gram TF-IDF + LogisticRegression model (feature f32)
- Trains a Gradient Boosting Machine with isotonic calibration (32 features)
- Runs 3-fold cross-validation
- Prints full metrics (F1, AUC, Recall, Precision, MCC, Brier, confusion matrix)
- Saves `models/scam_detector_gbm32.pkl` and `models/scam_detector_ngram.pkl`

**Expected output:**
```
[DATA] Loaded 5,574 messages — 747 spam / 4,827 ham
[SPLIT] Train: 4,459  Test: 1,115
[FEATURES] Extracting features for 4,459 messages...
[CV 3-fold]
  F1    : 0.9303 ± 0.0098
  AUC   : 0.9907 ± 0.0043
  Recall: 0.9047 ± 0.0177
...
[SAVE] models/scam_detector_gbm32.pkl
```

**Options:**
```bash
python train.py --data SMSSpamCollection --out models --cv 5 --test-size 0.20
```

---

## Step 3 — Test the Model

```bash
# Run demo predictions on preset messages
python predict.py --model-dir models

# Classify a single message
python predict.py --model-dir models --text "URGENT! Your account suspended. Verify at bit.ly/secure"

# Classify from a file (one message per line)
python predict.py --model-dir models --file my_messages.txt
```

**Expected output:**
```
🔴 [      SCAM] p=0.9843 lang=en
   URGENT! Your PayPal account suspended. Verify CVV at bit.ly/secure-verify now!
   ↑ url_shortener = 1.0
   ↑ has_sensitive = 1.0
   ↑ has_urgency = 1.0

🟢 [      SAFE] p=0.0312 lang=en
   Your OTP is 847291. Valid 10 min. Never share this code.
```

---

## Step 4 (Optional) — Multilingual Training

### 4a — Generate multilingual data
```bash
python generate_multilingual_data.py --n 400 --out multilingual_data.csv
# Generates 3,200 synthetic messages (400 scam + 400 safe × 4 languages)
```

### 4b — Train multilingual model
```bash
python train_multilingual.py \
  --sms SMSSpamCollection \
  --ml  multilingual_data.csv \
  --out models
```

**Saves:**
- `models/multilingual_scam_detector.pkl`  (~281 KB)
- `models/multilingual_ngram_model.pkl`    (~561 KB)
- `models/multilingual_results.json`

**Expected bundle size:** ~842 KB total — Android deployable.

---

## Step 5 (Optional) — Run the REST API

```bash
python api.py
# API running on http://localhost:5000
```

**Single prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"content": "URGENT! Verify your OTP at bit.ly/verify-now or account closes"}'
```

**Response:**
```json
{
  "verdict": "SCAM",
  "probability": 0.9921,
  "risk_pct": 99.2,
  "language": "en",
  "threshold": 0.9,
  "top_signals": [
    {"feature": "char_ngram_scam_score", "value": 0.9876},
    {"feature": "url_shortener", "value": 1.0},
    {"feature": "has_urgency", "value": 1.0}
  ],
  "latency_ms": 3.2
}
```

**Batch prediction:**
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"messages": ["msg1", "msg2", "msg3"]}'
```

---

## The 32 Features

| # | Feature | Type | Signal |
|---|---------|------|--------|
| f01 | has_urgency | binary | scam ▲ |
| f02 | has_money | binary | scam ▲ |
| f03 | has_sensitive | binary | scam ▲ |
| f04 | has_off_platform | binary | scam ▲ |
| f05 | has_threat | binary | scam ▲ |
| f06 | has_legitimacy_marker | binary | safe ▼ |
| f07 | text_length | int | context |
| f08 | exclamation_count | int | scam ▲ |
| f09 | question_count | int | context |
| f10 | uppercase_ratio | float | scam ▲ |
| f11 | digit_ratio | float | **dominant on real SMS** |
| f12 | char_entropy | float | context |
| f13 | avg_word_length | float | context |
| f14 | punctuation_density | float | context |
| f15 | urgency_density | float | scam ▲ |
| f16 | money_density | float | scam ▲ |
| f17 | sensitive_density | float | scam ▲ |
| f18 | num_urls | int | scam ▲ |
| f19 | url_density | float | scam ▲ |
| f20 | ip_url | binary | scam ▲ |
| f21 | url_shortener | binary | scam ▲ |
| f22 | risky_tld | binary | scam ▲ |
| f23 | domain_spoof | binary | scam ▲ |
| f24 | verified_domain | binary | safe ▼ |
| f25 | detected_lang_int | int | context |
| f26 | has_urgency_ml | binary | scam ▲ |
| f27 | has_money_ml | binary | scam ▲ |
| f28 | has_sensitive_ml | binary | scam ▲ |
| f29 | has_off_platform_ml | binary | scam ▲ |
| f30 | has_threat_ml | binary | scam ▲ |
| f31 | script_mismatch | float | scam ▲ |
| f32 | char_ngram_scam_score | float | scam ▲ |

---

## Model Architecture

```
Input message
      │
      ├─── Language detection (Unicode block analysis, <0.1ms)
      │
      ├─── Char 3-5gram TF-IDF + LogisticRegression → f32 score  (<2ms)
      │
      ├─── 32-feature extraction (f01–f32)                        (<1ms)
      │
      └─── CalibratedClassifierCV(GBM, isotonic)                  (<3ms)
                    │
                    └─── Probability → Language-specific threshold → Verdict
```

**Thresholds:**
| Language | Scam threshold |
|----------|---------------|
| English | 0.90 |
| Hindi | 0.85 |
| Telugu | 0.85 |
| Kannada | 0.85 |
| Marathi | 0.80 |

**Verdict zones:**
- `SAFE` — probability < 0.40
- `SUSPICIOUS` — 0.40 ≤ probability < threshold
- `SCAM` — probability ≥ threshold

---

## Expected Real-World Metrics (UCI SMS)

| Metric | Value |
|--------|-------|
| CV F1 (3-fold) | 0.9303 ± 0.0098 |
| CV AUC | 0.9907 ± 0.0043 |
| Test F1 | 0.9278 |
| Test AUC | 0.9907 |
| Test MCC | 0.9174 |
| Test Recall | 0.9060 |
| Test Precision | 0.9507 |
| False Positives | 7 / 1,115 |
| False Negatives | 14 / 1,115 |

---

## Dataset Notes

**UCI SMS Spam Collection (2012)**
- 5,574 messages: 747 spam, 4,827 ham
- Most spam = promotional/lottery, few URLs → `digit_ratio` is the dominant feature
- URL features (f18–f24) contribute near-zero importance on this corpus
- License: CC BY 4.0

**Why the synthetic F1 in the paper is higher:**  
Synthetic data has stronger URL patterns → URL features fire heavily → near-perfect separation.  
Real UCI SMS has almost no URLs in spam → only statistical/keyword features matter.  
Both are valid but measure different things. **F1=0.9303 on UCI SMS is the honest metric.**

---

## Troubleshooting

**`FileNotFoundError: SMSSpamCollection`**  
Make sure the file is in the same directory as `train.py`, or pass the full path:
```bash
python train.py --data /path/to/SMSSpamCollection
```

**`ModuleNotFoundError: No module named 'sklearn'`**
```bash
pip install -r requirements.txt
```

**Low F1 on your own data**  
The model was trained on UCI SMS (2012, English). For best results on newer phishing or non-English messages, add your own labeled data and retrain.

**API returns 500**  
Make sure models exist: `ls models/`. Run `python train.py` first.

---

## Citation

```
Adkine, V. (2026). ScamShield: An Interpretable Multi-Signal Scam Detection System.
DOI: 10.5281/zenodo.18988170
Dataset: Almeida, T.A. & Gómez Hidalgo, J.M. (2011). UCI SMS Spam Collection. DocEng 2011.
```
