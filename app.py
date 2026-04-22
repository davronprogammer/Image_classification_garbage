import json
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
CLASSES_PATH = BASE_DIR / "classes.json"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

model = None
classes = None
device = None
transform = None


def build_model(num_classes):
    classifier = models.efficientnet_b0(weights=None)
    classifier.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(classifier.classifier[1].in_features, num_classes),
    )
    return classifier


def load_model():
    """Load the trained model and image transform once."""
    global model, classes, device, transform

    if model is not None:
        return

    device = torch.device("cpu")

    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)

    loaded_model = build_model(len(classes))
    state_dict = torch.load(MODEL_PATH, map_location=device)
    loaded_model.load_state_dict(state_dict)
    loaded_model.to(device)
    loaded_model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    model = loaded_model
    print("Model loaded successfully")


def has_allowed_extension(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_file):
    try:
        load_model()

        image = Image.open(image_file).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        top_prob, top_idx = probs.max(0)
        predicted_class = classes[top_idx.item()]
        confidence = top_prob.item() * 100
        all_scores = {
            classes[i]: float((probs[i] * 100).item())
            for i in range(len(classes))
        }

        return {
            "success": True,
            "class": predicted_class,
            "confidence": round(confidence, 1),
            "all_scores": all_scores,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Prediction failed: {exc}",
        }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not has_allowed_extension(file.filename):
        return jsonify(
            {
                "success": False,
                "error": "Invalid file type. Allowed: JPG, PNG, GIF, WebP",
            }
        ), 400

    result = predict_image(file)
    status_code = 200 if result["success"] else 400
    return jsonify(result), status_code


@app.route("/health")
def health():
    return jsonify({"model_loaded": model is not None, "status": "ok"}), 200


if __name__ == "__main__":
    load_model()
    print("Server running on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
