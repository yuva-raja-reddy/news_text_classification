# News Text Classification 📰

A machine learning project that classifies news articles into different categories using Word2Vec and Random Forest classifier.

## Features

- Text classification into multiple news categories
- Word2Vec for text vectorization
- Random Forest classifier for prediction
- Interactive Gradio web interface
- Example news articles for testing
- Cross-validation and model evaluation

## Performance

- Accuracy: 97.53%
- Precision: 97.54%
- Recall: 97.53%
- F1 Score: 97.53%

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