"""
ScamShield — Multilingual Training Data Generator
Generates synthetic scam/safe messages in Hindi, Marathi, Telugu, Kannada.
Use this to supplement the UCI SMS dataset for multilingual training.

Output: multilingual_data.csv  (label, message, language)
"""

import random
import csv
import os

random.seed(42)

# ─────────────────────────────────────────────
# TEMPLATES  (scam and safe for each language)
# ─────────────────────────────────────────────

DATA = {
    "hi": {
        "scam": [
            "आपके बैंक खाते में संदिग्ध गतिविधि पाई गई है। तुरंत वेरीफाई करें: {url}",
            "बधाई हो! आपने {prize} की लॉटरी जीती है। अभी क्लेम करें: {url}",
            "आपका OTP है {otp}। इसे किसी के साथ साझा न करें।",  # actually safe (OTP)
            "SBI: आपका खाता बंद हो जाएगा। तुरंत {url} पर जाएं और वेरीफाई करें।",
            "आपके {bank} खाते से {amount} रुपये डेबिट हो गए। यदि नहीं किया तो {url} पर क्लिक करें।",
            "URGENT: आपका KYC अपडेट जरूरी है। 24 घंटे में न करने पर खाता फ्रीज होगा। {url}",
            "Congratulations! आपको {amount} रुपये का इनाम मिला है। व्हाट्सएप पर संपर्क करें: {phone}",
            "आपके पैन कार्ड पर {amount} रुपये का लोन अप्रूव हुआ। OTP बताएं: {url}",
        ],
        "safe": [
            "नमस्ते, कल की मीटिंग का समय बदल गया है। अब 3 बजे होगी।",
            "भाई, आज खाना खाने आ जाओ। मम्मी ने दाल बाटी बनाई है।",
            "आपका ऑर्डर {order_id} डिस्पैच हो गया है। डिलीवरी 2-3 दिन में होगी।",
            "आज का मौसम अच्छा है। शाम को पार्क जाएं।",
            "रिपोर्ट तैयार है। कृपया ईमेल चेक करें।",
            "धन्यवाद आपकी सेवा के लिए। हम जल्द संपर्क करेंगे।",
        ],
    },
    "mr": {
        "scam": [
            "आपल्या बँक खात्यात संशयास्पद क्रियाकलाप आढळला. ताबडतोब वेरिफाय करा: {url}",
            "अभिनंदन! आपण {prize} ची लॉटरी जिंकली आहे. आत्ताच क्लेम करा: {url}",
            "SBI: आपले खाते बंद होईल. ताबडतोब {url} वर जा आणि KYC अपडेट करा.",
            "आपल्या {bank} खात्यातून {amount} रुपये वजा झाले. {url} वर तपासा.",
            "URGENT: OTP सांगा नाहीतर खाते फ्रीज होईल: {url}",
            "तुम्हाला {amount} रुपये इनाम मिळाले आहे. व्हॉट्सॲपवर संपर्क करा: {phone}",
        ],
        "safe": [
            "नमस्कार, उद्याची बैठक रद्द करण्यात आली आहे.",
            "भाऊ, आज जेवायला ये. आई ने पुरणपोळी केली आहे.",
            "अहवाल तयार आहे. कृपया ईमेल तपासा.",
            "धन्यवाद! आपली सेवा उत्कृष्ट आहे.",
            "हवामान छान आहे. संध्याकाळी फिरायला जाऊया.",
        ],
    },
    "te": {
        "scam": [
            "మీ బ్యాంక్ ఖాతాలో అనుమానాస్పద కార్యకలాపం కనుగొనబడింది. వెంటనే వెరిఫై చేయండి: {url}",
            "అభినందనలు! మీరు {prize} లాటరీ గెలిచారు. ఇప్పుడు క్లెయిమ్ చేయండి: {url}",
            "SBI: మీ ఖాతా మూసివేయబడుతుంది. వెంటనే {url} వద్ద KYC అప్‌డేట్ చేయండి.",
            "మీ {bank} ఖాతా నుండి {amount} రూపాయలు డెబిట్ అయ్యాయి. {url} తనిఖీ చేయండి.",
            "URGENT: OTP చెప్పండి లేదా ఖాతా ఫ్రీజ్ అవుతుంది: {url}",
            "మీకు {amount} రూపాయల బహుమతి లభించింది. WhatsApp లో సంప్రదించండి: {phone}",
        ],
        "safe": [
            "నమస్కారం, రేపటి మీటింగ్ రద్దు చేయబడింది.",
            "అన్నా, ఈ రోజు భోజనానికి రా. అమ్మ పులిహోర చేసింది.",
            "నివేదిక సిద్ధంగా ఉంది. దయచేసి ఇమెయిల్ తనిఖీ చేయండి.",
            "ధన్యవాదాలు! మీ సేవ అద్భుతంగా ఉంది.",
            "వాతావరణం చాలా బాగుంది. సాయంత్రం నడకకు వెళ్దాం.",
        ],
    },
    "kn": {
        "scam": [
            "ನಿಮ್ಮ ಬ್ಯಾಂಕ್ ಖಾತೆಯಲ್ಲಿ ಸಂಶಯಾಸ್ಪದ ಚಟುವಟಿಕೆ ಕಂಡುಬಂದಿದೆ. ತಕ್ಷಣ ವೆರಿಫೈ ಮಾಡಿ: {url}",
            "ಅಭಿನಂದನೆಗಳು! ನೀವು {prize} ಲಾಟರಿ ಗೆದ್ದಿದ್ದೀರಿ. ಈಗಲೇ ಕ್ಲೇಮ್ ಮಾಡಿ: {url}",
            "SBI: ನಿಮ್ಮ ಖಾತೆ ಮುಚ್ಚಲಾಗುವುದು. ತಕ್ಷಣ {url} ನಲ್ಲಿ KYC ಅಪ್‌ಡೇಟ್ ಮಾಡಿ.",
            "ನಿಮ್ಮ {bank} ಖಾತೆಯಿಂದ {amount} ರೂಪಾಯಿ ಡೆಬಿಟ್ ಆಗಿದೆ. {url} ತಪಾಸಣೆ ಮಾಡಿ.",
            "URGENT: OTP ಹೇಳಿ ಇಲ್ಲವಾದಲ್ಲಿ ಖಾತೆ ಫ್ರೀಜ್ ಆಗುತ್ತದೆ: {url}",
            "ನಿಮಗೆ {amount} ರೂಪಾಯಿ ಬಹುಮಾನ ಸಿಕ್ಕಿದೆ. WhatsApp ನಲ್ಲಿ ಸಂಪರ್ಕಿಸಿ: {phone}",
        ],
        "safe": [
            "ನಮಸ್ಕಾರ, ನಾಳೆಯ ಸಭೆ ರದ್ದುಗೊಳಿಸಲಾಗಿದೆ.",
            "ಅಣ್ಣ, ಇಂದು ಊಟಕ್ಕೆ ಬಾ. ಅಮ್ಮ ಬಿಸಿಬೇಳೆ ಬಾತ್ ಮಾಡಿದ್ದಾರೆ.",
            "ವರದಿ ಸಿದ್ಧವಾಗಿದೆ. ದಯವಿಟ್ಟು ಇಮೇಲ್ ಪರಿಶೀಲಿಸಿ.",
            "ಧನ್ಯವಾದಗಳು! ನಿಮ್ಮ ಸೇವೆ ಅದ್ಭುತವಾಗಿದೆ.",
            "ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ. ಸಂಜೆ ವಾಯುವಿಹಾರಕ್ಕೆ ಹೋಗೋಣ.",
        ],
    },
}

URLS    = ["bit.ly/secure-verify","tinyurl.com/kyc-update","is.gd/bank-login","cutt.ly/verify-now"]
PRIZES  = ["₹50,00,000","₹10,00,000","$50,000","1 Crore"]
AMOUNTS = ["₹9,999","₹24,999","₹49,999","₹1,00,000"]
BANKS   = ["SBI","HDFC","ICICI","Axis","PNB"]
PHONES  = ["+91-9999999999","+91-8888888888","9876543210"]
ORDERS  = ["ORD-12345","ORD-67890"]
OTPS    = ["847291","123456","998877"]


def fill(template):
    return (template
        .replace("{url}",      random.choice(URLS))
        .replace("{prize}",    random.choice(PRIZES))
        .replace("{amount}",   random.choice(AMOUNTS))
        .replace("{bank}",     random.choice(BANKS))
        .replace("{phone}",    random.choice(PHONES))
        .replace("{order_id}", random.choice(ORDERS))
        .replace("{otp}",      random.choice(OTPS))
    )


def generate_dataset(n_per_class_per_lang=400, out_path="multilingual_data.csv"):
    rows = []
    for lang, templates in DATA.items():
        for _ in range(n_per_class_per_lang):
            msg = fill(random.choice(templates["scam"]))
            rows.append({"label":"spam","message":msg,"language":lang})
        for _ in range(n_per_class_per_lang):
            msg = fill(random.choice(templates["safe"]))
            rows.append({"label":"ham","message":msg,"language":lang})

    random.shuffle(rows)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label","message","language"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[GEN] Generated {len(rows):,} multilingual messages → {out_path}")
    counts = {}
    for r in rows:
        k = (r["language"], r["label"])
        counts[k] = counts.get(k, 0) + 1
    for (lang, lbl), cnt in sorted(counts.items()):
        print(f"      {lang}/{lbl}: {cnt:,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400,
                        help="Samples per class per language (default 400)")
    parser.add_argument("--out", default="multilingual_data.csv")
    args = parser.parse_args()
    generate_dataset(n_per_class_per_lang=args.n, out_path=args.out)
