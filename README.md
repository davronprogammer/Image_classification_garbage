# 🗑️ Garbage Classification Web Application

A production-grade web application for AI-powered waste classification using EfficientNet-B0 deep learning model.

## 📋 Features

- **🖼️ Image Upload**: Drag-and-drop or click-to-browse image upload
- **🎯 Real-time Prediction**: Fast inference on uploaded images
- **📊 Detailed Results**: Shows predicted class, confidence score, and all 6 classifications
- **🎨 Modern UI**: Dark eco-tech aesthetic with smooth animations
- **📱 Responsive Design**: Works on desktop and mobile devices
- **⚡ Optimized Backend**: Model loaded once at startup, efficient Flask routes
- **🛡️ Error Handling**: Graceful error messages for invalid files

## 🤖 Model Specifications

- **Architecture**: EfficientNet-B0 (transfer learning from ImageNet)
- **Classes**: 6 waste categories
  - 📦 Cardboard
  - 🍶 Glass
  - 🔩 Metal
  - 📄 Paper
  - 🧴 Plastic
  - 🗑️ Trash
- **Input**: 224×224 RGB images
- **Output**: Classification + confidence scores

## 📦 Project Structure

```
image_classification/
├── app.py                       # Flask backend application
├── best_model.pth               # Trained model weights
├── classes.json                 # Class labels
├── requirements.txt             # Python dependencies
├── templates/
│   └── index.html               # Frontend application
└── dataset/ (optional)          # Training data (not needed for inference)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

If you already have dependencies, you can install individually:
```bash
pip install Flask torch torchvision Pillow
```

### 2. Run the Application

```bash
python app.py
```

The server will start at: **http://127.0.0.1:5000**

### 3. Open in Browser

Navigate to `http://127.0.0.1:5000` and start classifying waste images!

## 💻 Backend API

### Endpoints

#### `GET /`
Serves the main application page.

**Response**: HTML page

---

#### `POST /predict`
Performs inference on uploaded image.

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: File field named `file`

**Expected File Types**: JPG, JPEG, PNG, GIF, WebP (max 16 MB)

**Response** (Success):
```json
{
  "success": true,
  "class": "plastic",
  "confidence": 95.3,
  "all_scores": {
    "cardboard": 2.1,
    "glass": 1.5,
    "metal": 0.8,
    "paper": 0.3,
    "plastic": 95.3,
    "trash": 0.0
  }
}
```

**Response** (Error):
```json
{
  "success": false,
  "error": "No file provided"
}
```

---

#### `GET /health`
Health check endpoint.

**Response**:
```json
{
  "status": "ok"
}
```

## 🎨 Frontend Features

### Upload
- **Drag & Drop**: Drag images onto the upload zone
- **Click to Browse**: Click to open file picker
- **Image Preview**: See selected image before classification

### Results Display
- **Class Emoji**: Unique emoji for each waste type
- **Confidence Bar**: Animated progress bar showing confidence percentage
- **Class Scores**: Bar chart showing prediction scores for all 6 classes
- **Error Handling**: Clear error messages for failed predictions

## ⚙️ Configuration

Edit these values in `app.py` to customize:

```python
# Maximum file upload size (default: 16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Server port (default: 5000)
# Change in the last line: app.run(..., port=5000)
```

## 🔧 Troubleshooting

### Model fails to load
- Ensure `best_model.pth` and `classes.json` exist in the same directory as `app.py`
- Verify PyTorch and torchvision are installed correctly
- Check disk space is available

### Port already in use
```bash
# Change port in app.py, last line:
# app.run(debug=True, host="127.0.0.1", port=5001)
```

### CUDA/GPU issues
- Application defaults to CPU if CUDA not available
- No changes needed - everything works on CPU (slightly slower inference)

### Image upload fails
- Check file format (JPG, PNG, GIF, WebP supported)
- Verify file size is under 16 MB
- Ensure image file is not corrupted

## 📊 Performance

- **Model Loading**: ~2-3 seconds on startup
- **Inference Time**: ~0.5-1 second per image (CPU)
- **Memory Usage**: ~500 MB

## 🔒 Security

- File size limit: 16 MB
- Only image formats accepted
- Server-side file validation
- No file persistence (images processed and discarded)

## 🎓 Model Training

The `best_model.pth` was trained using:
- Dataset: Garbage classification dataset (6 categories)
- Data split: 80% training, 20% validation
- Epochs: 10
- Optimizer: Adam (LR: 0.001)
- Data augmentation: flips, rotations, color jitter

For training code, see `main.py` (not required for inference).

## 📝 License

This project uses a pre-trained EfficientNet-B0 model from PyTorch/torchvision.

## 🤝 Support

For issues or questions about the application, check:
- Terminal output for error messages
- Browser console (F12) for frontend errors
- Flask debug mode provides detailed server logs

---

**Happy classifying!** ♻️
