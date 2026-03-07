# Emotion Classification Using Classical ML(FCNN), LSTM, and Transformers  
*A comparative study of NLP model families on multi‑class emotion detection*

---

## Overview  
This project builds an end‑to‑end **emotion classification system** that predicts six emotions from text data:

**sadness, joy, love, anger, fear, surprise**

The goal is to compare three major NLP modeling approaches:

1. **TF‑IDF + Fully Connected Neural Network (FCNN)**  
2. **Bidirectional LSTM with Embeddings**  
3. **DistilBERT Transformer (fine‑tuned)**  

Each model is trained, validated, and evaluated using **accuracy** **confusion matrices**.

---
'''
emotion-classification/
│
├── README.md                 # Project documentation
├── emotion_classification.ipynb   # Your single notebook
│
├── requirements.txt          # (Optional) Python dependencies
│
├── data/                     # (Optional) Raw or sample data
│   └── dataset.csv
│
└── images/                   # (Optional) Confusion matrices, plots
    ├── fcnn_cm.png
    ├── lstm_cm.png
    └── distilbert_cm.png
'''


---

## Key Features  
- Full preprocessing pipeline:  
  - lowercasing  
  - punctuation removal  
  - stopword removal  
  - lemmatization  
- Label encoding and class distribution analysis  
- TF‑IDF vectorization for classical models  
- Tokenization + padding for LSTM  
- HuggingFace tokenization for DistilBERT  
- Model comparison using:  
  - **Validation accuracy**  
  - **Confusion matrices**  

---

## 📊 Dataset  
The dataset contains short text samples labeled with one of six emotions.  
After cleaning and splitting:

- **80%** training  
- **20%** validation  
- Separate **test set** for final evaluation  

Class balance was checked to ensure fair evaluation.

---

## 🧼 Preprocessing  
A custom cleaning function was applied:

- Lowercase text  
- Remove punctuation  
- Tokenize  
- Remove stopwords  
- Lemmatize  
- Rejoin tokens  

This cleaned text is used for TF‑IDF and LSTM models.  
Transformers use the raw cleaned text without lemmatization or stopword removal.

---

## 🧪 Models

### 1️⃣ TF‑IDF + Fully Connected Neural Network  
- 5,000‑word TF‑IDF vocabulary  
- Two dense layers with dropout  
- Good baseline accuracy    
- Struggles with contextual understanding  

### 2️⃣ Bidirectional LSTM  
- Tokenized + padded sequences  
- Embedding layer (100‑dim)  
- BiLSTM (64 units)  
- Captures sequential patterns and context  

### 3️⃣ DistilBERT Transformer  
- Fine‑tuned using HuggingFace  
- Subword tokenization  
- Low learning rate (5e‑5)  
- Best overall performance   
- Significantly fewer misclassifications  

---

## 📈 Results

### ✔️ Validation Accuracy Comparison  
| Model         | Accuracy |
|---------------|----------|
| FCNN (TF‑IDF) |     |
| BiLSTM        |   |
| DistilBERT    |   | 

### ✔️ Confusion Matrices  
Confusion matrices were plotted for all models to analyze class‑wise performance.  
DistilBERT showed the cleanest diagonal and the least confusion between similar emotions.

---

## 🧩 Key Insights  
- Classical models struggle with nuanced emotional cues.  
- LSTMs improve performance by capturing sequential dependencies.  
- Transformers outperform both by leveraging deep contextual understanding.  
- Precision and recall reveal weaknesses that accuracy alone hides.  
- DistilBERT provides the most reliable and balanced predictions.



