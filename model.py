# model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def train_model(X_train, y_train):
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    evaluation_metrics = classification_report(y_test, y_pred)
    return evaluation_metrics

def predict_sentiment(model, text_samples):
    """
    Predict sentiment labels for a list of text samples using the trained model.

    Args:
        model: Trained machine learning model.
        text_samples (list): List of text samples.

    Returns:
        list: Predicted sentiment labels for each text sample.
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_samples_tfidf = tfidf_vectorizer.transform(text_samples)
    return model.predict(X_samples_tfidf)