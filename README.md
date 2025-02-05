# News Text Classification 📰

A machine learning project that classifies news articles into different categories using Word2Vec and Random Forest classifier.

## Features

- Text classification into multiple news categories
- Word2Vec for text vectorization
- Random Forest classifier for prediction
- Interactive Gradio web interface
- Example news articles for testing
- Cross-validation and model evaluation

## Model Performance

### CountVectorizer + Random Forest Model
- Accuracy: 97.08%
- Precision: 97.15%
- Recall: 97.08%
- F1 Score: 97.04%
```bash
Category-wise Performance:
               precision    recall  f1-score
     business       0.94      1.00      0.97
entertainment       0.97      0.90      0.93
     politics       0.97      1.00      0.98
        sport       0.98      1.00      0.99
         tech       1.00      0.94      0.97
```
### Word2Vec + Random Forest Model
- Accuracy: 96.18%
- Precision: 96.17%
- Recall: 96.18%
- F1 Score: 96.16%
```bash
Category-wise Performance:
               precision    recall  f1-score
     business       0.98      0.98      0.98
entertainment       0.93      0.90      0.91
     politics       0.94      0.98      0.96
        sport       0.99      0.99      0.99
         tech       0.95      0.95      0.95
```
## Installation

1. Clone the repository:
```bash
git clone https://github.com/yuva-raja-reddy/news_text_classification.git
cd news_text_classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the web interface:

```bash
python app.py
```

2. Enter news text or use example articles
3. Click "Predict News Category" to get classification

## Model Training

- The model was trained on the BBC News dataset using:
- Word2Vec for text vectorization
- Random Forest for classification
- 80-20 train-test split
- 6-fold cross-validation
