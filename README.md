# Lyrics Classification & Mood-Aware Music Search

An end-to-end NLP system for lyrics-based genre classification and mood-aware semantic music retrieval.
The project combines classical ML, transformer embeddings, and emotion modeling to build a hybrid search + classification pipeline over a large-scale lyrics dataset (~300K songs).

# Project details:

1. Lyrics Classification Pipeline
Genre classification from raw lyrics
Feature pipeline: TF-IDF + SVD + engineered style features
Models: Logistic Regression, Linear SVM, SGD, LightGBM

2. Mood-Aware Semantic Search
Sentence-BERT embeddings for semantic similarity
Transformer-based emotion inference
Valence-based mood scoring system
Hybrid ranking (semantic + emotion similarity)

3. Core Pipeline
Lyrics Data
   ↓
Preprocessing & Cleaning
   ↓
Feature Engineering
(TF-IDF + Style Features + SBERT Embeddings)
   ↓
Dimensionality Reduction (SVD)
   ↓
ML Classification Models
   ↓
Emotion Inference + Valence Mapping
   ↓
Hybrid Search Engine
(Semantic + Mood Ranking)

4. Models & Techniques
TF-IDF (n-gram based representation)
Sentence-BERT (MiniLM embeddings)
Transformer-based emotion classifier
SVD for dimensionality reduction
LightGBM / SVM / Logistic Regression / SGD

5. Key Capability
Given a text query or lyric snippet, the system can:
Predict genre category
Retrieve semantically similar songs
Align results based on emotional tone (mood matching)

6. Tech Stack
Python
PyTorch
HuggingFace Transformers
SentenceTransformers
Scikit-learn
LightGBM
NumPy / Pandas / SciPy

7. How to Run
pip install -r requirements.txt
    Classification pipeline : python lyrics_classification_ete_v1.py
    Mood-aware search : python mood_search_ete.py

*Notes: Built for large-scale lyrics dataset (~300K samples)
Originally developed in Colab, refactored into standalone scripts
GPU used for transformer inference where available.
