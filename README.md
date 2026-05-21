# Skin Disease Classification System 🔬

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)

## Overview
This repository contains my graduation project: an end-to-end Deep Learning system for classifying skin diseases. The project leverages an **EfficientNetV2** architecture to classify dermatological images into 6 distinct categories with high accuracy.

It includes the complete pipeline: from dataset exploration and model training (via Jupyter Notebooks) to a fully functional **REST API** using FastAPI.

### Supported Classes
1. Acne
2. Eczema
3. Fungal Infections
4. Melanoma
5. Psoriasis
6. Vitiligo

---

## Repository Structure

- `efficient_net_training.ipynb`: The primary notebook detailing data preprocessing, EfficientNetV2 training, Test-Time Augmentation (TTA), and evaluation.
- `api/`: Contains the FastAPI application (`app.py`), serving the trained `.h5` model as a REST endpoint.
- `api/test_api.py`: Automated integration test suite for the REST API.

*(Note: Datasets and trained `.h5` / `.keras` model weights are excluded from this repository due to GitHub size limits).*

---

## Features

- **High-Accuracy Classification**: Fine-tuned EfficientNetV2 architecture.
- **Test-Time Augmentation (TTA)**: Uses 7 augmented crops/flips during inference to dramatically improve confidence and accuracy.
- **REST API**: Production-ready FastAPI implementation with CORS, lazy-loading, and Swagger documentation.

---

## How to Run the API Locally

1. **Install dependencies:**
   ```bash
   cd api
   pip install -r requirements.txt
   ```

2. **Place the trained model:**
   Ensure your trained model is named `best_skin_model_effnet.h5` and you set the `SKIN_MODEL_PATH` environment variable to its location.

3. **Start the server:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   *The Swagger UI will be available at `http://localhost:8000/docs`.*

## License
This project was developed as a graduation requirement. All rights reserved.
