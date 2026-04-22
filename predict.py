import argparse
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def load_classes(path="classes.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    return model


def load_model(model_path="best_model.pth", classes_path="classes.json"):
    classes = load_classes(classes_path)
    device = torch.device("cpu")

    model = build_model(len(classes))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, classes, device


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(image_path):
    model, classes, device = load_model()
    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = probs.max(0)

    class_name = classes[top_idx.item()]
    confidence = top_prob.item() * 100

    print(f"Result: {class_name}")
    print(f"Confidence: {confidence:.1f}%")
    return class_name, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a waste image.")
    parser.add_argument("image", nargs="?", default="test.jpg", help="Path to an image file")
    args = parser.parse_args()
    predict(args.image)
