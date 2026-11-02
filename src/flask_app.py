"""Flask API wrapper based on program section 4.36.

Set model/tokenizer objects and preprocessing resources before creating the app.
Do not hard-code ngrok authentication tokens in source code.
"""
from flask import Flask, request, jsonify


def create_app(classification_function, classification_kwargs=None):
    app = Flask(__name__)
    classification_kwargs = classification_kwargs or {}

    @app.get("/")
    def home():
        return "Hate Speech Classification API is running."

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True) or {}
        sample_text = data.get("text", "")
        if not isinstance(sample_text, str) or not sample_text.strip():
            return jsonify({"error": "A non-empty 'text' field is required."}), 400
        results = classification_function(sample_text, **classification_kwargs)
        return jsonify(results)

    return app


if __name__ == "__main__":
    raise SystemExit("Import create_app() and provide the loaded models/tokenizers.")
