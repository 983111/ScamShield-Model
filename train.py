"""
ScamShield — Full Training Script
Dataset: UCI SMS Spam Collection (public, CC BY 4.0)
Download: https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
"""

import pandas as pd
import numpy as np
import joblib
import json
import re
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, recall_score, precision_score,
    matthews_corrcoef, brier_score_loss, confusion_matrix,
    classification_report
)

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
def load_data(path="SMSSpamCollection"):
    """
    Loads the UCI SMS Spam Collection tab-separated file.
    Columns: label (ham/spam), message
    """
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["label", "message"], encoding="latin-1"
    )
    df["label_int"] = (df["label"] == "spam").astype(int)
    print(f"[DATA] Loaded {len(df):,} messages — "
          f"{df['label_int'].sum():,} spam / {(df['label_int']==0).sum():,} ham")
    return df

# ─────────────────────────────────────────────
# 2. FEATURE EXTRACTION  (24 features)
# ─────────────────────────────────────────────

# Keyword dictionaries
URGENCY_WORDS = [
    "urgent", "immediately", "asap", "right now", "expires",
    "act now", "last chance", "final notice", "within 24", "within 48",
    "deadline", "hurry", "don't delay", "do not ignore", "respond now"
]
MONEY_WORDS = [
    "won", "winner", "prize", "lottery", "cash", "free money", "reward",
    "jackpot", "earn", "income", "profit", "investment", "bitcoin", "crypto",
    "guaranteed", "100%", "£", "$", "€", "free gift", "claim now", "bonus"
]
SENSITIVE_WORDS = [
    "password", "pin", "cvv", "otp", "one-time", "social security",
    "bank account", "credit card", "debit card", "verify your", "confirm your",
    "update your", "account number", "routing number", "ssn"
]
OFF_PLATFORM_WORDS = [
    "telegram", "whatsapp", "signal", "dm me", "text me", "call this number",
    "contact us at", "reach us on", "add me on", "message me on"
]
THREAT_WORDS = [
    "will be suspended", "will be deleted", "will be blocked", "will be closed",
    "account terminated", "security alert", "breach detected", "unauthorized access",
    "immediate action required", "failure to respond"
]
LEGITIMACY_MARKERS = [
    "regards", "sincerely", "thank you for", "documentation", "meeting",
    "schedule", "attached", "as discussed", "per our conversation",
    "please find", "report", "invoice", "receipt"
]
URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "is.gd", "cutt.ly", "t.co",
    "ow.ly", "buff.ly", "short.link", "rb.gy", "bl.ink", "tiny.cc"
]
VERIFIED_DOMAINS = [
    "google.com", "apple.com", "amazon.com", "microsoft.com", "github.com",
    "paypal.com", "bank.com", "gov.uk", "gov.in", "flipkart.com", "hdfc.com"
]
RISKY_TLDS = [".tk", ".ml", ".ga", ".cf", ".pw", ".xyz", ".top", ".click", ".loan"]


def char_entropy(text):
    """Shannon entropy of character distribution."""
    if not text:
        return 0.0
    counts = {}
    for c in text:
        counts[c] = counts.get(c, 0) + 1
    total = len(text)
    return -sum((v/total) * np.log2(v/total) for v in counts.values())


def extract_urls(text):
    return re.findall(r'https?://\S+|www\.\S+|[a-z0-9.-]+\.[a-z]{2,6}(?:/\S*)?', text, re.I)


def extract_features(text):
    """
    Returns a 24-element feature vector for a single message.
    f01 has_urgency          f02 has_money
    f03 has_sensitive        f04 has_off_platform
    f05 has_threat           f06 has_legitimacy_marker
    f07 text_length          f08 exclamation_count
    f09 question_count       f10 uppercase_ratio
    f11 digit_ratio          f12 char_entropy
    f13 avg_word_length      f14 punctuation_density
    f15 urgency_density      f16 money_density
    f17 sensitive_density    f18 num_urls
    f19 url_density          f20 ip_url
    f21 url_shortener        f22 risky_tld
    f23 domain_spoof         f24 verified_domain
    """
    tl = text.lower()
    urls = extract_urls(tl)
    words = tl.split()

    # Binary keyword signals (f01–f06)
    has_urgency          = int(any(k in tl for k in URGENCY_WORDS))
    has_money            = int(any(k in tl for k in MONEY_WORDS))
    has_sensitive        = int(any(k in tl for k in SENSITIVE_WORDS))
    has_off_platform     = int(any(k in tl for k in OFF_PLATFORM_WORDS))
    has_threat           = int(any(k in tl for k in THREAT_WORDS))
    has_legitimacy_marker= int(any(k in tl for k in LEGITIMACY_MARKERS))

    # Statistical features (f07–f14)
    text_length      = len(text)
    exclamation_count= text.count('!')
    question_count   = text.count('?')
    uppercase_ratio  = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    digit_ratio      = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    entropy          = char_entropy(text)
    avg_word_length  = np.mean([len(w) for w in words]) if words else 0
    punct_density    = sum(1 for c in text if c in '.,;:!?-()[]{}') / max(len(text), 1)

    # Keyword density (f15–f17)
    urgency_density  = sum(1 for k in URGENCY_WORDS  if k in tl) / max(len(words), 1)
    money_density    = sum(1 for k in MONEY_WORDS     if k in tl) / max(len(words), 1)
    sensitive_density= sum(1 for k in SENSITIVE_WORDS if k in tl) / max(len(words), 1)

    # URL features (f18–f24)
    num_urls       = len(urls)
    url_density    = num_urls / max(len(words), 1)
    ip_url         = int(bool(re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', tl)))
    url_shortener  = int(any(s in tl for s in URL_SHORTENERS))
    risky_tld      = int(any(tld in tl for tld in RISKY_TLDS))
    domain_spoof   = int(bool(re.search(
        r'(paypa1|g00gle|amaz0n|micros0ft|app1e|netfl1x|bankofamer1ca)', tl
    )))
    verified_domain= int(any(d in tl for d in VERIFIED_DOMAINS))

    return [
        has_urgency, has_money, has_sensitive, has_off_platform,
        has_threat, has_legitimacy_marker,
        text_length, exclamation_count, question_count, uppercase_ratio,
        digit_ratio, entropy, avg_word_length, punct_density,
        urgency_density, money_density, sensitive_density,
        num_urls, url_density, ip_url, url_shortener,
        risky_tld, domain_spoof, verified_domain
    ]


FEATURE_NAMES = [
    "has_urgency","has_money","has_sensitive","has_off_platform",
    "has_threat","has_legitimacy_marker",
    "text_length","exclamation_count","question_count","uppercase_ratio",
    "digit_ratio","char_entropy","avg_word_length","punctuation_density",
    "urgency_density","money_density","sensitive_density",
    "num_urls","url_density","ip_url","url_shortener",
    "risky_tld","domain_spoof","verified_domain"
]


def build_feature_matrix(messages):
    print(f"[FEATURES] Extracting features for {len(messages):,} messages...")
    X = np.array([extract_features(m) for m in messages], dtype=float)
    print(f"[FEATURES] Matrix shape: {X.shape}")
    return X

# ─────────────────────────────────────────────
# 3. CHAR N-GRAM MODEL  (for feature f32 / multilingual)
# ─────────────────────────────────────────────

def build_ngram_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=8000,
            sublinear_tf=True,
        )),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            C=1.0, solver="liblinear",
            class_weight="balanced",
            max_iter=1000
        )),
    ])

# ─────────────────────────────────────────────
# 4. MAIN GBM MODEL
# ─────────────────────────────────────────────

def build_gbm_model():
    base = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42
    )
    return CalibratedClassifierCV(base, method="isotonic", cv=3)

# ─────────────────────────────────────────────
# 5. EVALUATION HELPERS
# ─────────────────────────────────────────────

def evaluate(y_true, y_pred, y_prob, label=""):
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    rec = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    bri = brier_score_loss(y_true, y_prob)
    cm  = confusion_matrix(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  F1        : {f1:.4f}")
    print(f"  AUC       : {auc:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  Precision : {pre:.4f}")
    print(f"  MCC       : {mcc:.4f}")
    print(f"  Brier     : {bri:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print(classification_report(y_true, y_pred, target_names=["ham","spam"]))
    return {"f1":f1,"auc":auc,"recall":rec,"precision":pre,"mcc":mcc,"brier":bri,
            "tn":int(cm[0,0]),"fp":int(cm[0,1]),"fn":int(cm[1,0]),"tp":int(cm[1,1])}


def cross_validate_model(model, X, y, cv=3):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = cross_validate(
        model, X, y,
        cv=skf,
        scoring=["f1","roc_auc","recall"],
        return_train_score=False,
        n_jobs=-1
    )
    print(f"\n[CV {cv}-fold]")
    print(f"  F1    : {results['test_f1'].mean():.4f} ± {results['test_f1'].std():.4f}")
    print(f"  AUC   : {results['test_roc_auc'].mean():.4f} ± {results['test_roc_auc'].std():.4f}")
    print(f"  Recall: {results['test_recall'].mean():.4f} ± {results['test_recall'].std():.4f}")
    return results

# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def print_feature_importance(model, feature_names):
    """Works for CalibratedClassifierCV wrapping GBM."""
    try:
        estimators = model.calibrated_classifiers_
        importances = np.mean(
            [e.estimator.feature_importances_ for e in estimators], axis=0
        )
        ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        print("\n[FEATURE IMPORTANCE]")
        for name, imp in ranked:
            bar = "█" * int(imp * 400)
            print(f"  {name:<25} {imp:.4f}  {bar}")
        return dict(ranked)
    except Exception as e:
        print(f"[WARN] Could not extract feature importance: {e}")
        return {}

# ─────────────────────────────────────────────
# 7. MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ScamShield Training")
    parser.add_argument("--data", default="SMSSpamCollection",
                        help="Path to UCI SMS Spam Collection tab file")
    parser.add_argument("--out", default="models",
                        help="Output directory for saved models")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── Load ──
    df = load_data(args.data)

    # ── Stratified 80/20 split (same as paper) ──
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(
        df, test_size=args.test_size,
        stratify=df["label_int"], random_state=42
    )
    print(f"[SPLIT] Train: {len(df_train):,}  Test: {len(df_test):,}")

    y_train = df_train["label_int"].values
    y_test  = df_test["label_int"].values

    # ── Char n-gram model ──
    print("\n[NGRAM] Training char 3-5gram model...")
    ngram_model = build_ngram_model()
    ngram_model.fit(df_train["message"].tolist(), y_train)

    # ── Re-extract the FULL 32-feature matrix (must match predict.py exactly) ──
    # Import here to avoid circular import at module level
    import importlib, sys
    # predict.py must be in the same folder
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    pred_mod = importlib.import_module("predict")

    def build_32(messages, ngram_mdl):
        rows = []
        for msg in messages:
            feats, _ = pred_mod.extract_features_32(msg, ngram_mdl)
            rows.append(feats)
        return np.array(rows, dtype=float)

    print("[FEATURES] Building 32-feature train matrix...")
    X_train_32 = build_32(df_train["message"].tolist(), ngram_model)
    print("[FEATURES] Building 32-feature test matrix...")
    X_test_32  = build_32(df_test["message"].tolist(),  ngram_model)
    feature_names_32 = pred_mod.FEATURE_NAMES_32
    print(f"[FEATURES] 32-feat shapes — train:{X_train_32.shape}  test:{X_test_32.shape}")

    # ── Cross-validation on 32-feature matrix ──
    print(f"\n[CV] Running {args.cv}-fold cross-validation on train set...")
    gbm_for_cv = build_gbm_model()
    cv_results = cross_validate_model(gbm_for_cv, X_train_32, y_train, cv=args.cv)

    # ── Final training on full train set ──
    print("\n[TRAIN] Fitting final model on full training set...")
    gbm_model = build_gbm_model()
    gbm_model.fit(X_train_32, y_train)

    # ── Test set evaluation ──
    y_pred = gbm_model.predict(X_test_32)
    y_prob = gbm_model.predict_proba(X_test_32)[:,1]
    test_metrics = evaluate(y_test, y_pred, y_prob, label="GBM + Char N-gram (Test Set)")

    # ── Feature importance ──
    fi = print_feature_importance(gbm_model, feature_names_32)

    # ── Save models ──
    gbm_path   = os.path.join(args.out, "scam_detector_gbm32.pkl")
    ngram_path = os.path.join(args.out, "scam_detector_ngram.pkl")
    joblib.dump(gbm_model,   gbm_path,   compress=3)
    joblib.dump(ngram_model, ngram_path, compress=3)
    print(f"\n[SAVE] GBM model    → {gbm_path}  ({os.path.getsize(gbm_path)/1024:.0f} KB)")
    print(f"[SAVE] N-gram model → {ngram_path} ({os.path.getsize(ngram_path)/1024:.0f} KB)")

    # ── Save metrics JSON ──
    results = {
        "cv": {
            "f1_mean":  float(cv_results["test_f1"].mean()),
            "f1_std":   float(cv_results["test_f1"].std()),
            "auc_mean": float(cv_results["test_roc_auc"].mean()),
            "auc_std":  float(cv_results["test_roc_auc"].std()),
        },
        "test": test_metrics,
        "feature_importance": fi,
        "feature_names": feature_names_32,
    }
    metrics_path = os.path.join(args.out, "training_results.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] Metrics      → {metrics_path}")
    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()