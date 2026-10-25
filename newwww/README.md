# 🫁 PulmoScan AI • Pneumonia Detection & Clinical UI

A pediatric chest radiograph screening system using an Anti-Overfitting Convolutional Neural Network with interactive clinical web interfaces.

---

## 📁 Project Contents (`newwww/`)

| File | Description |
| :--- | :--- |
| **`app.py`** | **Streamlit Clinical Web Dashboard**: Modern UI with real-time inference, Grad-CAM attention heatmap, threshold slider, patient demographics, and printable report export. |
| **`web_app.py`** | **Zero-Dependency Web Console**: Lightweight local Python server (runs with just `python web_app.py`, open `http://localhost:5000`). |
| **`pneumonia.ipynb`** | **Trained Google Colab Notebook**: Completed training run achieving **87.5% Test Accuracy**, **0.9371 ROC-AUC**, and Youden's J optimal cutoff of **0.7414**. |
| **`requirements.txt`** | Dependency list (`streamlit`, `tensorflow`, `scikit-learn`, `pillow`, `matplotlib`). |

---

## 🚀 How to Launch the UI

### Option 1: Launch the Streamlit Clinical Dashboard *(Recommended)*
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser at `http://localhost:8501`.

---

### Option 2: Launch the Zero-Configuration Web App
No complex installations needed — uses Python's standard web server:
1. Run:
   ```bash
   python web_app.py
   ```
2. Open your browser at `http://localhost:5000`.

---

## 🎯 Model & Classification Details
* **Trained Weights**: Copy `best_pneumonia_model.keras` (generated from `pneumonia.ipynb`) into this folder.
* **Calibrated Decision Cutoff**: `0.7414` (calculated via Youden's J index $J = \text{TPR} - \text{FPR}$).
* **Overfitting Mitigations**: Conv-BatchNorm variance shift eliminated, output bias initialized to class prior, label smoothing ($0.08$), and exposure-invariant data augmentation.
