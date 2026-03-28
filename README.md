# Emotion Classification Using Classical ML(FCNN), LSTM, and Transformers  
*A comparative study of NLP model families on multi‑class emotion detection*
---

## Overview  
This project builds an end‑to‑end **emotion classification system** that predicts six emotions from text data

**sadness, joy, love, anger, fear, surprise**

The goal is to compare three major NLP modeling approaches

1. **TF‑IDF + Fully Connected Neural Network (FCNN)**  
2. **Bidirectional LSTM with Embeddings**  
3. **DistilBERT Transformer (fine‑tuned)**  

Each model is trained, validated, and evaluated using **accuracy** **confusion matrices**
---
```
Project-Structure/
│
├── data/                          # text data as train and test file in .txt
│   ├── test.txt
│   └── train.txt
│
├── README.md                      # Project documentation
├── emotion_classification.ipynb   # Main notebook for training and evaluation models 
└── requirements.txt               # Python dependencies
```
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
## Dataset  
The dataset contains short text samples labeled with one of six emotions.  
After cleaning and splitting

- **80%** training  
- **20%** validation  
- Splitted into separate **validation set** for final evaluation  

Class balance was checked to ensure fair evaluation.

---

## Preprocessing  
A custom cleaning function was applied

- Lowercase text  
- Remove punctuation  
- Tokenize  
- Remove stopwords  
- Lemmatize  
- Rejoin tokens  

This cleaned text is used for TF‑IDF and LSTM models.  
Transformers use the raw cleaned text without lemmatization or stopword removal.

---

## Models

### Model 1: TF‑IDF with Fully Connected Neural Network(FCNN)
- 5,000‑word TF‑IDF vocabulary  
- Two dense layers with dropout  
- Good baseline accuracy    
- Struggles with contextual understanding  

### Model 2:  Bidirectional LSTM  
- Tokenized + padded sequences  
- Embedding layer (100‑dim)  
- BiLSTM (64 units)  
- Captures sequential patterns and context  

### Model 3:  DistilBERT Transformer  
- Fine‑tuned using HuggingFace  
- Subword tokenization  
- Low learning rate (5e‑5)  
- Best overall performance   
- Significantly fewer misclassifications  

---

## Results

### Validation Accuracy Comparison  
| Model         | Validation Accuracy |
|---------------|---------------------|
| FCNN (TF‑IDF) |      ~ 0.8618       |
| BiLSTM        |      ~ 0.8633       |
| DistilBERT    |        0.9332       | 

### Confusion Matrices  
Confusion matrices were plotted for all models to analyze class‑wise performance  
DistilBERT showed the cleanest diagonal and the least confusion between similar emotions

---

## Key Insights and highlights of Pre-trained model
- Classical models struggle with nuanced emotional cues  
- LSTMs improve performance by capturing sequential dependencies
- Transformers outperform both by leveraging deep contextual understanding and sequential dependencies  
- DistilBERT provides the most reliable and balanced predictions



