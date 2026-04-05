"""
ScamShield — Inference Script
Load trained models and predict on new messages.
"""

import joblib
import numpy as np
import re
import os
import sys

# ── Import feature extraction from train.py ──
sys.path.insert(0, os.path.dirname(__file__))
from train import (
    extract_features, FEATURE_NAMES,
    URL_SHORTENERS, VERIFIED_DOMAINS, RISKY_TLDS,
    URGENCY_WORDS, MONEY_WORDS, SENSITIVE_WORDS,
    OFF_PLATFORM_WORDS, THREAT_WORDS
)

# ─────────────────────────────────────────────
# MULTILINGUAL LANGUAGE DETECTION
# ─────────────────────────────────────────────

MARATHI_MARKERS = ["आहे", "करा", "आणि", "होईल", "नाही", "करून", "आत्ताच", "ताबडतोब"]
HINDI_MARKERS   = ["है", "करें", "और", "होगा", "नहीं", "तुरंत", "अभी", "हो जाएगा"]

ML_LEXICONS = {
    "hi": {
        "URGENCY":      ["तुरंत","अभी","खाता बंद","jaldi","turant","otp bhejo"],
        "MONEY":        ["लॉटरी","पैसे","lottery","paisa","inaam","prize"],
        "SENSITIVE":    ["पासवर्ड","ओटीपी","otp","password","pin","khata"],
        "OFF_PLATFORM": ["व्हाट्सएप","whatsapp","telegram","call karo"],
        "THREAT":       ["बंद हो जाएगा","suspend","block","band ho","service band"],
    },
    "mr": {
        "URGENCY":      ["ताबडतोब","आत्ताच","खाते बंद","otp sanga","lakar"],
        "MONEY":        ["लॉटरी","पैसे","inaam","prize","lottery"],
        "SENSITIVE":    ["पासवर्ड","ओटीपी","otp","password","pin"],
        "OFF_PLATFORM": ["व्हाट्सएप","whatsapp","telegram"],
        "THREAT":       ["बंद होईल","suspend","block","seva band"],
    },
    "te": {
        "URGENCY":      ["వెంటనే","ఇప్పుడే","ventane","ippude","urgent"],
        "MONEY":        ["లాటరీ","ఓటీపీ","prize","lottery","paisa"],
        "SENSITIVE":    ["పాస్వర్డ్","ఓటీపీ","password","otp","pin"],
        "OFF_PLATFORM": ["టెలిగ్రామ్","whatsapp","telegram"],
        "THREAT":       ["ఖాతా మూసివేయబడుతుంది","suspend","block"],
    },
    "kn": {
        "URGENCY":      ["ತಕ್ಷಣ","ಈಗಲೇ","takshana","igale","urgent"],
        "MONEY":        ["ಲಾಟರಿ","ಒಟಿಪಿ","prize","lottery","paisa"],
        "SENSITIVE":    ["ಪಾಸ್‌ವರ್ಡ್","ಒಟಿಪಿ","password","otp","pin"],
        "OFF_PLATFORM": ["ವಾಟ್ಸಾಪ್","whatsapp","telegram"],
        "THREAT":       ["ಖಾತೆ ಮುಚ್ಚಲಾಗುವುದು","suspend","block"],
    },
    "en": {
        "URGENCY":      URGENCY_WORDS,
        "MONEY":        MONEY_WORDS,
        "SENSITIVE":    SENSITIVE_WORDS,
        "OFF_PLATFORM": OFF_PLATFORM_WORDS,
        "THREAT":       THREAT_WORDS,
    },
}

LANG_INT = {"en":0, "hi":1, "mr":2, "te":3, "kn":4, "other":5}
LANG_THRESHOLD = {"en":0.90, "hi":0.85, "mr":0.80, "te":0.85, "kn":0.85, "other":0.85}


def strip_urls(text):
    return re.sub(r'https?://\S+|www\.\S+|[a-z0-9.-]+\.[a-z]{2,6}(?:/\S*)?', ' ', text, flags=re.I)


def count_script_chars(text):
    counts = {
        "devanagari": len(re.findall(r'[\u0900-\u097F]', text)),
        "telugu":     len(re.findall(r'[\u0C00-\u0C7F]', text)),
        "kannada":    len(re.findall(r'[\u0C80-\u0CFF]', text)),
        "latin":      len(re.findall(r'[a-zA-Z]', text)),
    }
    return counts


def detect_language(text):
    stripped = strip_urls(text)
    counts = count_script_chars(stripped)
    dominant = max(counts, key=counts.get)
    if counts[dominant] == 0:
        return "other"
    if dominant == "devanagari":
        return "mr" if any(w in text for w in MARATHI_MARKERS) else "hi"
    return {"telugu":"te","kannada":"kn","latin":"en"}.get(dominant, "other")


def script_mismatch_score(text):
    stripped = strip_urls(text)
    counts = count_script_chars(stripped)
    native = counts["devanagari"] + counts["telugu"] + counts["kannada"]
    latin  = counts["latin"]
    if native == 0:
        return 0.0
    return min(native, latin) / (native + latin)


def has_ml_keyword(lang, category, text):
    tl = text.lower()
    lexicon = ML_LEXICONS.get(lang, ML_LEXICONS["en"])
    return int(any(k.lower() in tl for k in lexicon.get(category, [])))


def extract_features_32(text, ngram_model=None):
    """Returns a 32-element feature vector."""
    # f01–f24: original English features
    base = extract_features(text)

    # f25: language int
    lang = detect_language(text)
    f25  = LANG_INT.get(lang, 5)

    # f26–f30: multilingual keyword signals
    f26 = has_ml_keyword(lang, "URGENCY",      text)
    f27 = has_ml_keyword(lang, "MONEY",        text)
    f28 = has_ml_keyword(lang, "SENSITIVE",    text)
    f29 = has_ml_keyword(lang, "OFF_PLATFORM", text)
    f30 = has_ml_keyword(lang, "THREAT",       text)

    # f31: script mismatch
    f31 = script_mismatch_score(text)

    # f32: char n-gram score
    if ngram_model is not None:
        try:
            f32 = float(ngram_model.predict_proba([text])[0][1])
        except Exception:
            f32 = 0.0
    else:
        f32 = 0.0

    return base + [f25, f26, f27, f28, f29, f30, f31, f32], lang


FEATURE_NAMES_32 = FEATURE_NAMES + [
    "detected_lang_int","has_urgency_ml","has_money_ml","has_sensitive_ml",
    "has_off_platform_ml","has_threat_ml","script_mismatch","char_ngram_scam_score"
]


# ─────────────────────────────────────────────
# PREDICTOR CLASS
# ─────────────────────────────────────────────

class ScamShield:
    def __init__(self, model_dir="models"):
        gbm_path   = os.path.join(model_dir, "scam_detector_gbm32.pkl")
        ngram_path = os.path.join(model_dir, "scam_detector_ngram.pkl")

        if not os.path.exists(gbm_path):
            raise FileNotFoundError(f"Model not found: {gbm_path}\nRun train.py first.")

        self.gbm_model   = joblib.load(gbm_path)
        self.ngram_model = joblib.load(ngram_path) if os.path.exists(ngram_path) else None
        print(f"[LOAD] GBM model loaded from {gbm_path}")
        if self.ngram_model:
            print(f"[LOAD] N-gram model loaded from {ngram_path}")
        else:
            print("[WARN] N-gram model not found. f32 will be 0.0.")

    def predict(self, text: str) -> dict:
        features, lang = extract_features_32(text, self.ngram_model)
        X = [features]
        prob  = float(self.gbm_model.predict_proba(X)[0][1])
        threshold = LANG_THRESHOLD.get(lang, 0.85)

        if prob >= threshold:
            verdict = "SCAM"
        elif prob >= 0.40:
            verdict = "SUSPICIOUS"
        else:
            verdict = "SAFE"

        # Top signals
        top = sorted(
            zip(FEATURE_NAMES_32, features),
            key=lambda x: abs(x[1]), reverse=True
        )[:5]

        return {
            "verdict":         verdict,
            "probability":     round(prob, 4),
            "risk_pct":        round(prob * 100, 1),
            "language":        lang,
            "threshold":       threshold,
            "top_signals":     [{"feature": n, "value": round(v,4)} for n,v in top],
        }

    def predict_batch(self, texts: list) -> list:
        return [self.predict(t) for t in texts]


# ─────────────────────────────────────────────
# CLI DEMO
# ─────────────────────────────────────────────

def main():
    import argparse, json
    parser = argparse.ArgumentParser(description="ScamShield Inference")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--text", type=str, help="Single message to classify")
    parser.add_argument("--file", type=str, help="Text file with one message per line")
    args = parser.parse_args()

    shield = ScamShield(model_dir=args.model_dir)

    demo_messages = [
        # English scam
        "URGENT! Your PayPal account suspended. Verify CVV at bit.ly/secure-verify now!",
        # Hindi scam
        "आपके बैंक खाते में संदिग्ध गतिविधि पाई गई है। तुरंत वेरीफाई करें: bit.ly/bank-verify",
        # Telugu scam
        "మీ బ్యాంక్ ఖాతాలో అనుమానాస్పద కార్యకలాపం. వెంటనే వెరిఫై చేయండి: bit.ly/verify",
        # Kannada safe
        "ಅಣ್ಣ ನಾಳೆ ಬರ್ತೀಯಾ? ಒಟ್ಟಿಗೆ ಊಟ ಮಾಡೋಣ",
        # Legit OTP
        "Your OTP is 847291. Valid 10 min. Never share this code.",
        # Lottery
        "Congratulations! You've WON £1,000,000 in the National Lottery! Claim: bit.ly/claim123",
    ]

    if args.text:
        result = shield.predict(args.text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.file:
        with open(args.file) as f:
            messages = [line.strip() for line in f if line.strip()]
        results = shield.predict_batch(messages)
        for msg, res in zip(messages, results):
            print(f"[{res['verdict']:>10}] {msg[:80]}")
        return

    # Default: run demo messages
    print("\n" + "="*70)
    print("  ScamShield Demo Predictions")
    print("="*70)
    for msg in demo_messages:
        res = shield.predict(msg)
        flag = {"SCAM":"🔴","SUSPICIOUS":"🟡","SAFE":"🟢"}[res["verdict"]]
        print(f"\n{flag} [{res['verdict']:>10}] p={res['probability']:.4f} lang={res['language']}")
        print(f"   {msg[:90]}")
        for sig in res["top_signals"][:3]:
            if sig["value"] > 0:
                print(f"   ↑ {sig['feature']} = {sig['value']}")


if __name__ == "__main__":
    main()
