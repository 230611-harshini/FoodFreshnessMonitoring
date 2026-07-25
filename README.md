#  Food Freshness Monitoring System

An AI-powered Food Freshness Monitoring System that classifies food items as **Fresh** or **Spoiled** using Deep Learning. The application also provides an estimated shelf-life and includes a user-friendly web interface for image upload and prediction.

---

## 📌 Features

- Fresh vs Spoiled food classification
- Supports multiple food categories:
  - 🍞 Bread
  - 🥛 Dairy
  - 🍎 Fruits
  - 🥦 Vegetables
- Shelf-life estimation
- FastAPI backend
- Simple HTML, CSS, and JavaScript frontend
- Pretrained EfficientNet model using PyTorch
- Real-time image prediction

---

## 🛠️ Tech Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- FastAPI
- Python

### Deep Learning
- PyTorch
- Torchvision
- EfficientNet-B0

### Other Libraries
- NumPy
- Pillow
- Matplotlib
- Scikit-learn

---

### Dataset
https://www.kaggle.com/datasets/maheen00shahid/fresh-and-spoiled-food-image-dataset

---

## 📂 Project Structure

```
FoodFreshnessMonitoring/
│
├── backend/
│   ├── app.py
│   ├── predictor.py
│   └── shelf_life.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── training/
│   └── train.py
│
├── dataset/
├── uploads/
├── outputs/
├── models/
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/230611-harshini/FoodFreshnessMonitoring.git
```

### Move into the project

```bash
cd FoodFreshnessMonitoring
```

### Create virtual environment

```bash
python -m venv venv
```

### Activate virtual environment

Windows

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Backend

```bash
python -m uvicorn backend.app:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

## ▶️ Run the Frontend

```bash
cd frontend
python -m http.server 5500
```

Open:

```
http://127.0.0.1:5500
```

---

## 📊 Model

- EfficientNet-B0 (Pretrained)
- Transfer Learning
- Image Size: 224 × 224
- Optimizer: Adam
- Loss Function: CrossEntropyLoss

---

## 📷 How It Works

1. Upload a food image.
2. The model predicts whether it is fresh or spoiled.
3. The application displays:
   - Predicted class
   - Estimated shelf-life

---

## 📌 Future Improvements

- Support more food categories
- Mobile application
- Real-time camera prediction
- Barcode scanning
- Cloud deployment

---

## 👩‍💻 Author

**Harshini S**

B.Tech Artificial Intelligence & Data Science

RMK Engineering College

GitHub: https://github.com/230611-harshini
