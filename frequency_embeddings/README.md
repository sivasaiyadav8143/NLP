# 📘 Frequency-Based Word Embeddings in NLP

This repository contains hands-on implementations of foundational word embedding techniques — Bag-of-Words, N-grams, and TF-IDF — as introduced in the article [Word Embeddings in NLP: From Bag-of-Words to Transformers (Part 1)](https://medium.com/p/4688627a728f).

These frequency-based methods are the building blocks of modern NLP pipelines. They help transform raw text into numerical representations that can be used for classification, search, and clustering.

---

## 📂 Repository Structure

```
frequency_embeddings/
├── Bag-of-Words/
│   ├── bow_support_ticket_classifier.ipynb
│   ├── README.md
│   └── requirements.txt
├── N-grams/
│   ├── ngrams_hotel_review.ipynb
│   ├── ngrams_hotel_review.md
│   └── requirements.txt
└── TF-IDF/
    ├── tfidf_doc_search.ipynb
    └── requirements.txt
```
---

## 🔍 Modules Overview

### 🧾 Bag-of-Words
- **Use case**: Classify support tickets based on keyword frequency.
- **Highlights**: Sparse vector representation, vocabulary size control, basic preprocessing.

### 🧾 N-grams
- **Use case**: Analyze hotel reviews using bigrams and trigrams.
- **Highlights**: Phrase-level context, token sequencing, n-gram generation.

### 🧾 TF-IDF
- **Use case**: Document search based on term importance.
- **Highlights**: Term weighting, inverse document frequency, cosine similarity.

---

## 🛠️ Setup

```bash
# Create environment
python -m venv freq_embed_env
source freq_embed_env/bin/activate

# Install dependencies
pip install -r requirements.txt  # Run inside each module folder
```

📚 Related Reading
This repo complements the concepts discussed in:

📄 Medium Article  
Word Embeddings in NLP: From Bag-of-Words to Transformers (Part 1) [Medium](https://medium.com/@sivasai-yadav)
Explore how frequency-based methods laid the groundwork for semantic embeddings like Word2Vec, GloVe, and BERT.

🤝 Contributions
Feel free to fork, improve, or extend the notebooks with your own datasets or embedding techniques. Pull requests welcome!
