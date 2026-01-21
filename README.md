# Support Ticket Classifier using Bag-of-Words

An automated support ticket routing system that classifies customer tickets into **Billing**, **Technical**, or **Sales** categories using simple yet effective Bag-of-Words (BoW) representation and Logistic Regression.

This project demonstrates how basic NLP techniques can solve real-world business problems with high accuracy and complete interpretability.

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Dataset](#dataset)
- [Key Learnings](#key-learnings)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

## 🎯 Overview

This project implements a ticket classification system that automatically routes customer support tickets to the appropriate team based on the ticket content. It uses **Bag-of-Words (BoW)** for text representation and **Logistic Regression** for classification.

**Why this matters:**
- Reduces manual ticket routing time
- Ensures tickets reach the right team faster
- Improves customer satisfaction through quicker response times
- Provides interpretable results (you can see which keywords drive decisions)

## 🔍 Problem Statement

Support teams receive hundreds of tickets daily across multiple categories:
- **Billing**: Payment issues, refunds, invoices
- **Technical**: Bugs, errors, crashes, system issues
- **Sales**: Product demos, pricing inquiries, upgrades

Manual routing is:
- ⏱️ Time-consuming
- ❌ Error-prone
- 📉 Not scalable as ticket volume grows

**Goal**: Build an automated system that accurately classifies tickets based on their text content.

## 💡 Solution Approach

### Why Bag-of-Words?

For this use case, BoW is ideal because:

1. **Distinct Vocabularies**: Each category has clear keyword patterns
   - Billing: "refund," "payment," "charge," "invoice"
   - Technical: "error," "bug," "crash," "timeout"
   - Sales: "demo," "pricing," "upgrade," "enterprise"

2. **Speed**: Real-time classification with minimal computational overhead

3. **Interpretability**: Support managers can see exactly why tickets are routed

4. **Simplicity**: Easy to maintain and debug

### Methodology

1. **Text Vectorization**: Convert ticket text to numerical vectors using BoW
2. **Feature Engineering**: Remove stop words, convert to lowercase
3. **Model Training**: Logistic Regression classifier
4. **Evaluation**: Accuracy, precision, recall, F1-score per category
5. **Deployment**: Save model for production use

## 📊 Dataset

### Sample Size
- **Total Tickets**: 45
- **Billing**: 15 tickets
- **Technical**: 15 tickets
- **Sales**: 15 tickets

### Sample Tickets

**Billing:**
```
"I was charged twice for my subscription this month"
"Need a refund for the payment made yesterday"
"Can I get an invoice for last month's payment"
```

**Technical:**
```
"The app keeps crashing when I try to login"
"Getting error 404 when accessing the dashboard"
"Software bug: cannot save my work"
```

**Sales:**
```
"I'm interested in the enterprise pricing plan"
"Can I schedule a product demo for next week"
"What features are included in the premium package"
```

### Data Split
- **Training Set**: 80% (36 tickets)
- **Test Set**: 20% (9 tickets)


## 🎓 Key Learnings

### What Works Well

✅ **Simple keyword-based tasks**: When categories have distinct vocabularies, BoW excels

✅ **Fast training and inference**: No complex computations, perfect for production

✅ **Interpretability**: Stakeholders can understand exactly why decisions are made

✅ **Low computational requirements**: Can run on modest hardware

### When to Use Bag-of-Words

- Clear keyword patterns exist between categories
- Speed and interpretability are priorities
- You need a quick baseline
- Dataset is small to medium-sized
- Word order doesn't significantly impact meaning for your task

### When NOT to Use Bag-of-Words

❌ **Nuanced sentiment analysis**: Can't distinguish "not good" from "good"

❌ **Semantic similarity**: Doesn't know "excellent" and "great" are related

❌ **Context-dependent meaning**: Can't handle words with multiple meanings

❌ **Complex language understanding**: Sarcasm, idioms, subtle meaning

## ⚠️ Limitations

1. **No Semantic Understanding**
   - Treats "good" and "excellent" as completely different words
   - Doesn't understand synonyms or related concepts

2. **Word Order Ignored**
   - "not good" and "good" are treated similarly
   - Negations can be missed

3. **Sparse Vectors**
   - Most values are zero
   - Memory inefficient for large vocabularies

4. **No Context Awareness**
   - Can't distinguish different meanings of the same word
   - Example: "bank" (financial) vs "bank" (river)

5. **Vocabulary Dependent**
   - New words not in training vocabulary are ignored
   - Typos or variations can cause issues

## 🔮 Future Improvements

### Short-term
- [ ] Add bigrams to capture phrases like "not good"
- [ ] Implement TF-IDF weighting for better feature importance
- [ ] Expand dataset to 500+ tickets per category
- [ ] Add confidence threshold for "uncertain" routing

### Medium-term
- [ ] Compare with Word2Vec embeddings
- [ ] Implement active learning for continuous improvement
- [ ] Add multi-label classification (tickets can be both Billing + Technical)
- [ ] Build web API for real-time classification

### Long-term
- [ ] Transition to BERT for complex cases
- [ ] Implement priority scoring based on ticket urgency
- [ ] Add automated response suggestions
- [ ] Build dashboard for ticket analytics


## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Siva Sai Yadav**
- LinkedIn: [sivasai-yadav](www.linkedin.com/in/sivasai-mudugandla)
- Medium: [@sivasai-yadav](https://medium.com/@sivasai-yadav)

## 🙏 Acknowledgments

- Part of my **Word Embeddings in NLP** article series
- Inspired by real-world customer support automation needs
- Built during my Applied Gen AI learning journey

## 📚 Related Resources

### My Article Series
- [Part 1: Frequency-Based Embeddings (BoW, TF-IDF, N-grams)](#) - This project
- [Part 2: Neural Embeddings (Word2Vec, GloVe, FastText)](#) - Coming soon
- [Part 3: Contextual Embeddings (BERT, GPT)](#) - Coming soon
---

⭐ **If you found this project helpful, please give it a star!**

📧 **Questions?** Feel free to open an issue or reach out directly.

---

*Last updated: January 2026*
