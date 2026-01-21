# Support Ticket Classifier using Bag-of-Words

An automated support ticket routing system that classifies customer tickets into **Billing**, **Technical**, or **Sales** categories using simple yet effective Bag-of-Words (BoW) representation and Logistic Regression.

This project demonstrates how basic NLP techniques can solve real-world business problems with high accuracy and complete interpretability.

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Learnings](#key-learnings)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
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

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | ~95-100% |
| **Training Time** | < 1 second |
| **Prediction Time** | < 10ms per ticket |

### Per-Category Performance

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Billing | 1.00 | 1.00 | 1.00 |
| Technical | 1.00 | 1.00 | 1.00 |
| Sales | 1.00 | 1.00 | 1.00 |

### Top Keywords per Category

**Billing Team:**
- `refund`, `payment`, `charged`, `invoice`, `subscription`, `billing`

**Technical Team:**
- `error`, `crash`, `bug`, `timeout`, `server`, `ssl`, `database`

**Sales Team:**
- `pricing`, `demo`, `upgrade`, `enterprise`, `premium`, `features`

### Confusion Matrix

The model achieves near-perfect classification with minimal confusion between categories.

![Confusion Matrix](outputs/confusion_matrix_bow.png)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
```

## 💻 Usage

### Option 1: Run the Jupyter Notebook

```bash
jupyter notebook bow_support_ticket_classifier.ipynb
```

Execute all cells to:
1. Load the dataset
2. Train the model
3. Evaluate performance
4. Test with new tickets
5. Visualize results

### Option 2: Use the Saved Model

```python
import pickle

# Load the saved model
with open('bow_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('ticket_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Classify a new ticket
new_ticket = ["I need a refund for my last payment"]
ticket_vector = vectorizer.transform(new_ticket)
prediction = classifier.predict(ticket_vector)
confidence = classifier.predict_proba(ticket_vector)

print(f"Route to: {prediction[0]} team")
print(f"Confidence: {max(confidence[0]):.2%}")
```

### Option 3: Quick Prediction Function

```python
def classify_ticket(ticket_text, vectorizer, classifier):
    """
    Classify a support ticket into Billing, Technical, or Sales.
    
    Args:
        ticket_text (str): The ticket content
        vectorizer: Trained CountVectorizer
        classifier: Trained classifier model
    
    Returns:
        dict: Prediction results with category and confidence
    """
    # Vectorize the ticket
    ticket_vector = vectorizer.transform([ticket_text])
    
    # Predict
    prediction = classifier.predict(ticket_vector)[0]
    probabilities = classifier.predict_proba(ticket_vector)[0]
    
    # Map probabilities to categories
    categories = ['Billing', 'Sales', 'Technical']
    confidence_scores = dict(zip(categories, probabilities))
    
    return {
        'category': prediction,
        'confidence': max(probabilities),
        'all_scores': confidence_scores
    }

# Example usage
result = classify_ticket(
    "The app crashed and I lost my data",
    vectorizer,
    classifier
)

print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All scores: {result['all_scores']}")
```

## 📁 Project Structure

```
bow-support-ticket-classifier/
├── bow_support_ticket_classifier.ipynb  # Main notebook
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
├── bow_vectorizer.pkl                    # Saved vectorizer
├── ticket_classifier.pkl                 # Saved classifier model
├── outputs/
│   ├── confusion_matrix_bow.png         # Confusion matrix visualization
│   └── word_frequencies_bow.png         # Word frequency chart
└── data/
    └── sample_tickets.csv               # (Optional) Export of sample data
```

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

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Issues**: Found a bug? Open an issue
2. **Suggest Features**: Have ideas for improvements? Let's discuss
3. **Submit PRs**: 
   - Fork the repository
   - Create a feature branch (`git checkout -b feature/AmazingFeature`)
   - Commit your changes (`git commit -m 'Add AmazingFeature'`)
   - Push to the branch (`git push origin feature/AmazingFeature`)
   - Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Siva Sai Yadav**

- GitHub: [@sivasai-yadav](https://github.com/sivasai-yadav)
- LinkedIn: [sivasai-yadav](https://linkedin.com/in/sivasai-yadav)
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

### Learn More
- [Scikit-learn CountVectorizer Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [Logistic Regression for Text Classification](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Understanding Bag-of-Words Model](https://en.wikipedia.org/wiki/Bag-of-words_model)

---

⭐ **If you found this project helpful, please give it a star!**

📧 **Questions?** Feel free to open an issue or reach out directly.

---

*Last updated: January 2026*
