"""Flask app serving the fine-tuned PEGASUS AG News headline generator."""
import json
import os

from flask import Flask, jsonify, render_template, request

from model_utils import STRATEGIES, HeadlineGenerator

app = Flask(__name__)

MIN_CHARS, MAX_CHARS = 40, 3000

generator = HeadlineGenerator.from_env()

with open(os.path.join(app.root_path, "results.json")) as f:
    RESULTS = json.load(f)


@app.get("/")
def index():
    return render_template(
        "index.html",
        strategies={k: v["label"] for k, v in STRATEGIES.items()},
        mock=generator.mock,
    )


@app.post("/api/generate")
def generate():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    strategy = body.get("strategy", "beam")

    if len(text) < MIN_CHARS:
        return jsonify(error=f"Paste at least {MIN_CHARS} characters of article text."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Keep the article under {MAX_CHARS} characters."), 400
    if strategy not in STRATEGIES:
        return jsonify(error=f"Unknown decoding strategy '{strategy}'."), 400

    try:
        return jsonify(generator.generate(text, strategy))
    except Exception as exc:  # surface model errors to the UI instead of a 500 page
        app.logger.exception("Generation failed")
        return jsonify(error=f"Generation failed: {exc}"), 500


@app.get("/api/results")
def results():
    return jsonify(RESULTS)


@app.get("/health")
def health():
    return jsonify(status="ok", model=generator.model_id, device=generator.device,
                   mock=generator.mock)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
