# 🔮 PrismLens AI — Real-Time Adaptive Color Detection

## 📋 Problem Statement
Improve color detection of images with different resolutions & color levels, processing up to **20-30 images in < 1 second** with **≥ 95% accuracy**.

## 💡 Solution
Lightweight CNN (TFLite) + K-Means RGB Analysis + Color Temperature + Color Cast Detection. 
- **12 color classes**
- **96-98% accuracy**
- **Sub-300ms inference** per image

## ✨ Features
- 🧠 **Lightweight CNN**: 158K params, 98% accuracy on 12 color classes
- ⚡ **Ultra-Fast**: ~300ms per image (30×30 K-Means optimization)
- 🌡️ **Color Temperature**: Detects Warm/Neutral/Cool lighting (Kelvin scale)
- 💡 **Color Cast Detection**: Identifies lighting color casts
- 📐 **RGB Ratio Analysis**: Exact Red/Green/Blue percentages
- 🎨 **Color Palette**: Top 5 dominant colors extracted
- 📦 **Edge-Ready**: TFLite model (~500KB)
- 🌐 **REST API**: FastAPI with batch processing
- 🖥️ **Web Frontend**: Drag & drop UI with real-time results

---

## 📁 Project Structure

```text
COLOR_DETECTION_PRISM/
├── .fetch-client/
├── .venv/
├── .venv311/
├── frontend/
│   └── index.html
├── src/
│   ├── .venv311/
│   ├── data/
│   │   ├── processed/
│   │   └── raw/
│   ├── models/
│   │   ├── best_model.h5
│   │   ├── color_model.h5
│   │   ├── color_model.tflite
│   │   └── labels.txt
│   ├── api.py
│   ├── generate_all_colors.py
│   └── test_results/
│       ├── formats/
│       ├── test_images/
│       ├── accuracy_results.json
│       ├── final_summary.json
│       ├── single_prediction_results.json
│       ├── stress_test.jpg
│       └── test_summary.json
├── README.md
└── requirements.txt

🚀 Setup & Installation

# 1. Create virtual environment
python -m venv .venv311

# 2. Activate virtual environment
# Windows:
.venv311\Scripts\activate
# Mac/Linux:
source .venv311/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train model (if not already trained)
python src/generate_all_colors.py

# 5. Start API server
python src/api.py

# 6. Start frontend (open a new terminal)
cd frontend
python -m http.server 3000

# 7. Open browser
# Navigate to: http://localhost:3000

🌐 API Endpoints

Method	Endpoint	Description
GET	/	Health check
POST	/predict	Single image color detection
POST	/predict_batch	Batch processing (multiple images)
GET	/docs	Swagger UI


📤 API Response (JSON)

{
  "cnn_prediction": {
    "color": "blue",
    "confidence": 95.5
  },
  "rgb_analysis": {
    "rgb_value": [30, 76, 148],
    "hex": "#1e4c94",
    "ratios": {"red": 15.0, "green": 30.0, "blue": 55.0},
    "color_palette": [...]
  },
  "color_temperature": {
    "temperature_name": "Cool (Overcast)",
    "kelvin": 6200
  },
  "color_cast": {
    "cast_color": "blue",
    "cast_strength": 18.5,
    "has_cast": true
  }
}


🛠️ Technologies

🧪 TensorFlow/Keras — CNN training

📱 TensorFlow Lite — Edge deployment

🖼️ OpenCV — Image processing & K-Means

⚡ FastAPI — REST API

🌍 HTML/CSS/JS — Frontend UI