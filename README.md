# Meme Sentiment Analyzer - English + Roman Urdu

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/DL-TensorFlow%2FKeras-orange)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-yellow)](https://scikit-learn.org)

---

## 📌 Overview

**Meme Sentiment Analyzer** is a multimodal machine-learning web application that classifies the **sentiment and tone** of internet memes. Given an uploaded meme image, the system simultaneously predicts **five different sentiment dimensions**:

| Dimension        | Classes |
|------------------|---------|
| 🎭 **Humour**     | Funny / Sarcastic / Not Funny |
| 😏 **Sarcasm**    | Sarcastic / Not Sarcastic |
| 🚫 **Offensive**  | Offensive / Not Offensive |
| 💪 **Motivational** | Motivational / Not Motivational |
| 🌐 **Overall Sentiment** | Positive / Negative / Neutral |

Memes are a unique, increasingly widespread form of digital communication that blend **visual content with text** — making meme understanding a challenging and novel problem in multimodal machine learning. This project tackles it by combining deep learning image features with natural language processing on the text embedded in memes.

The system supports memes written in **English and Roman Urdu**, as OCR is used to extract any text embedded in the meme image before analysis.

---

## 🧠 How It Works

The system implements a **multimodal pipeline** that fuses visual and textual signals:

### Step 1 — OCR Text Extraction
When a meme image is uploaded, `pytesseract` (an OCR engine) reads and extracts any text embedded in the meme. This works for both **English** and **Roman Urdu** text. The extracted text is then cleaned (lowercased, non-alphanumeric characters removed).

### Step 2 — Image Feature Extraction (ResNet50)
A **ResNet50** deep convolutional neural network, pre-trained on ImageNet (via TensorFlow/Keras), acts as a visual feature extractor. The top classification layer is removed and the penultimate layer outputs a **2048-dimensional feature vector** that captures rich semantic information from the meme image.

### Step 3 — Text Vectorization (TF-IDF)
The OCR-extracted text is converted into numerical features using a **TF-IDF Vectorizer** (top 4000 unigrams and bigrams, with English stop-word removal). This captures the statistical importance of words and phrases in the meme text.

### Step 4 — Feature Fusion
The TF-IDF text features and ResNet50 image features are **horizontally stacked** into a single combined feature vector using sparse matrices — giving the model both visual and linguistic understanding.

### Step 5 — Sentiment Classification (5 Models)
Five separate classifiers — one per sentiment dimension — predict the labels:

| Label | Model | Feature Engineering |
|-------|-------|---------------------|
| Humour | SGD Classifier | Original + Nystroem non-linear features |
| Sarcastic | SGD (elasticnet, ultra-low alpha) | Original + Nystroem non-linear features |
| Offensive | SGD (elasticnet, ultra-low alpha) | Original + Nystroem non-linear features |
| Motivational | SGD Classifier | Original + Nystroem non-linear features |
| Overall Sentiment | Random Forest (200 trees, depth=30) | Original features only |

**Nystroem Kernel Approximation** is applied to help the linear SGD classifier learn non-linear patterns by generating 500 synthetic kernel features.

### Step 6 — Flask Web Interface
A lightweight **Flask** app allows users to upload a meme image via a browser and receive instant sentiment predictions across all five dimensions.

---

## 🗂️ Repository Structure

```
Meme-Sentiment-Analyzer---English-Roman-Urdu/
│
├── app.py                     # Flask web application (entry point)
├── train.py                   # Full model training pipeline
│
├── models/                    # Saved trained model checkpoints
│   ├── vectorizer.pkl         # Trained TF-IDF vectorizer
│   ├── nystroem_transformer.pkl  # Fitted Nystroem transformer
│   ├── humour_model.pkl
│   ├── sarcastic_model.pkl
│   ├── offensive_model.pkl
│   ├── motivational_model.pkl
│   └── overall_model.pkl
│
├── static/                    # Static assets (CSS, JS, uploaded images)
├── templates/                 # HTML templates for the Flask UI
│
├── testImages/                # Sample meme images for testing
├── train_cleaned_final.csv    # Cleaned and labelled training dataset
├── confusion_matrices/        # Confusion matrix plots from model evaluation
│
├── .gitignore                 # Git exclusions
└── README.md                  # This file
```

> **Note:** `image_features_resnet.npy` (~109 MB) is excluded from this repository due to GitHub's file size limit. It is auto-generated by running `train.py`.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Tesseract OCR installed on your system
  - **macOS:** `brew install tesseract`
  - **Ubuntu:** `sudo apt install tesseract-ocr`
  - **Windows:** [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ArslanKamran/Meme-Sentiment-Analyzer---English-Roman-Urdu.git
   cd Meme-Sentiment-Analyzer---English-Roman-Urdu
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate     # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask tensorflow pillow pytesseract numpy scipy scikit-learn tqdm requests matplotlib
   ```

---

## 🚀 Usage

### Run the Web App
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000`, upload a meme image, and click **Analyse** to see predictions across all five sentiment dimensions.

> **⚠️ Note:** The `models/` directory must exist with all trained `.pkl` files before running `app.py`. Train the models first if they are missing.

### Train the Models from Scratch
```bash
python train.py
```
This will:
1. Load and clean `train_cleaned_final.csv`
2. Download meme images from URLs and extract ResNet50 features (cached to `image_features_resnet.npy`)
3. Vectorize OCR text with TF-IDF
4. Apply Nystroem kernel approximation
5. Train five classifiers and save them under `models/`
6. Generate confusion matrix plots under `confusion_matrices/`

---

## 🌐 Language Support

| Language     | Status                                                   |
|--------------|----------------------------------------------------------|
| English      | ✅ Fully supported via OCR text extraction               |
| Roman Urdu   | ✅ Supported — OCR reads Roman Urdu text embedded in memes |

> The system uses OCR to extract any text from the meme image, which means it naturally handles both English and Roman Urdu text that appears within the meme.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: description"`
4. Push: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Arslan Kamran**  
FCCU — CSCS460 Machine Learning | 5th Semester
