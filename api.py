"""
ScamShield — Flask REST API
Serves predictions over HTTP. Works for all 5 languages.
"""

from flask import Flask, request, jsonify
from predict import ScamShield
import os
import time

app = Flask(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
shield = ScamShield(model_dir=MODEL_DIR)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ScamShield"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: {"content": "message text here"}
    Returns: verdict, probability, language, top_signals
    """
    data = request.get_json(force=True, silent=True)
    if not data or "content" not in data:
        return jsonify({"error": "Missing 'content' field"}), 400

    text = str(data["content"]).strip()
    if not text:
        return jsonify({"error": "Empty message"}), 400
    if len(text) > 10000:
        return jsonify({"error": "Message too long (max 10000 chars)"}), 400

    t0  = time.perf_counter()
    res = shield.predict(text)
    ms  = round((time.perf_counter() - t0) * 1000, 2)
    res["latency_ms"] = ms

    return jsonify(res)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    POST /predict/batch
    Body: {"messages": ["msg1", "msg2", ...]}
    """
    data = request.get_json(force=True, silent=True)
    if not data or "messages" not in data:
        return jsonify({"error": "Missing 'messages' field"}), 400

    messages = data["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        return jsonify({"error": "'messages' must be a non-empty list"}), 400
    if len(messages) > 100:
        return jsonify({"error": "Max 100 messages per batch"}), 400

    t0 = time.perf_counter()
    results = shield.predict_batch([str(m) for m in messages])
    ms = round((time.perf_counter() - t0) * 1000, 2)

    return jsonify({"results": results, "count": len(results), "latency_ms": ms})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[API] ScamShield API running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
