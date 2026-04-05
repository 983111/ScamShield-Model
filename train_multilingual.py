"""
ScamShield — Multilingual Training Script
Trains on: UCI SMS Spam (English) + generated multilingual data

Run generate_multilingual_data.py first, then:
  python train_multilingual.py --sms SMSSpamCollection --ml multilingual_data.csv
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, roc_auc_score, recall_score, precision_score,
    matthews_corrcoef, brier_score_loss, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(__file__))
from train   import build_gbm_model, build_ngram_model, build_feature_matrix, FEATURE_NAMES
from predict import extract_features_32, FEATURE_NAMES_32, detect_language, LANG_THRESHOLD


def load_combined(sms_path, ml_path):
    # English UCI SMS
    df_en = pd.read_csv(
        sms_path, sep="\t", header=None,
        names=["label","message"], encoding="latin-1"
    )
    df_en["language"] = "en"
    df_en["label_int"] = (df_en["label"] == "spam").astype(int)
    print(f"[DATA] UCI SMS: {len(df_en):,} messages")

    # Multilingual
    df_ml = pd.read_csv(ml_path, encoding="utf-8")
    df_ml["label_int"] = (df_ml["label"] == "spam").astype(int)
    print(f"[DATA] Multilingual: {len(df_ml):,} messages")

    df = pd.concat([
        df_en[["label","message","language","label_int"]],
        df_ml[["label","message","language","label_int"]]
    ], ignore_index=True).sample(frac=1, random_state=42)

    print(f"[DATA] Combined: {len(df):,} total — "
          f"{df['label_int'].sum():,} spam / {(df['label_int']==0).sum():,} ham")
    return df


def build_feature_matrix_32(messages, ngram_model):
    rows = []
    for msg in messages:
        feats, _ = extract_features_32(msg, ngram_model)
        rows.append(feats)
    return np.array(rows, dtype=float)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sms", default="SMSSpamCollection")
    parser.add_argument("--ml",  default="multilingual_data.csv")
    parser.add_argument("--out", default="models")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv",  type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_combined(args.sms, args.ml)

    df_train, df_test = train_test_split(
        df, test_size=args.test_size,
        stratify=df["label_int"], random_state=42
    )
    print(f"[SPLIT] Train: {len(df_train):,}  Test: {len(df_test):,}")

    # Train char n-gram model first
    print("\n[NGRAM] Training char n-gram model...")
    ngram_model = build_ngram_model()
    ngram_model.fit(df_train["message"].tolist(), df_train["label_int"].values)

    # Build 32-feature matrices
    print("[FEATURES] Extracting 32-feature matrices...")
    X_train = build_feature_matrix_32(df_train["message"].tolist(), ngram_model)
    X_test  = build_feature_matrix_32(df_test["message"].tolist(),  ngram_model)
    y_train = df_train["label_int"].values
    y_test  = df_test["label_int"].values
    print(f"[FEATURES] Train: {X_train.shape}  Test: {X_test.shape}")

    # Cross-validation
    print(f"\n[CV] {args.cv}-fold CV on train set...")
    model_cv = build_gbm_model()
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    cv_res = cross_validate(
        model_cv, X_train, y_train, cv=skf,
        scoring=["f1","roc_auc","recall"], n_jobs=-1
    )
    print(f"  F1    : {cv_res['test_f1'].mean():.4f} ± {cv_res['test_f1'].std():.4f}")
    print(f"  AUC   : {cv_res['test_roc_auc'].mean():.4f} ± {cv_res['test_roc_auc'].std():.4f}")
    print(f"  Recall: {cv_res['test_recall'].mean():.4f} ± {cv_res['test_recall'].std():.4f}")

    # Final training
    print("\n[TRAIN] Fitting final multilingual model...")
    model = build_gbm_model()
    model.fit(X_train, y_train)

    # Test evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    rec = recall_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    print(f"\n[TEST] F1={f1:.4f}  AUC={auc:.4f}  Recall={rec:.4f}  Precision={pre:.4f}  MCC={mcc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["ham","spam"]))
    print(f"  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    # Per-language breakdown
    print("\n[PER LANGUAGE]")
    df_test_copy = df_test.copy().reset_index(drop=True)
    df_test_copy["pred"] = y_pred
    df_test_copy["prob"] = y_prob
    for lang in ["en","hi","mr","te","kn"]:
        sub = df_test_copy[df_test_copy["language"]==lang]
        if len(sub) == 0:
            continue
        try:
            lf1  = f1_score(sub["label_int"], sub["pred"])
            lauc = roc_auc_score(sub["label_int"], sub["prob"])
            print(f"  {lang}: n={len(sub):4d}  F1={lf1:.4f}  AUC={lauc:.4f}")
        except Exception as e:
            print(f"  {lang}: n={len(sub):4d}  (could not compute — {e})")

    # Save
    gbm_path   = os.path.join(args.out, "multilingual_scam_detector.pkl")
    ngram_path = os.path.join(args.out, "multilingual_ngram_model.pkl")
    joblib.dump(model,       gbm_path,   compress=3)
    joblib.dump(ngram_model, ngram_path, compress=3)
    print(f"\n[SAVE] {gbm_path}  ({os.path.getsize(gbm_path)/1024:.0f} KB)")
    print(f"[SAVE] {ngram_path}  ({os.path.getsize(ngram_path)/1024:.0f} KB)")

    results = {
        "cv": {
            "f1_mean":  float(cv_res["test_f1"].mean()),
            "f1_std":   float(cv_res["test_f1"].std()),
            "auc_mean": float(cv_res["test_roc_auc"].mean()),
        },
        "test": {"f1":f1,"auc":auc,"recall":rec,"precision":pre,"mcc":mcc,
                 "tn":int(cm[0,0]),"fp":int(cm[0,1]),"fn":int(cm[1,0]),"tp":int(cm[1,1])},
    }
    with open(os.path.join(args.out,"multilingual_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n✓ Multilingual training complete.")


if __name__ == "__main__":
    main()
