import pickle
import numpy as np

# Load the model and vectorizer
def load_sentiment_model():
    with open("models/comment_sentiments.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("models/transformed.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return clf, vectorizer

# Predict sentiment for a single review
def predict_sentiment(review_text, clf, vectorizer):
    # Transform the review text
    review_vector = vectorizer.transform([review_text])
    # Predict
    prediction = clf.predict(review_vector)[0]
    # Return 'Good' or 'Bad' (adjust based on your model's labeling)
    return 'Good' if prediction == 1 else 'Bad'

# Example usage
if __name__ == "__main__":
    # Load model once
    clf, vectorizer = load_sentiment_model()
    
    # Test reviews
    test_reviews = [
        "This movie was amazing! I loved every minute of it.",
        "Terrible film, complete waste of time.",
        "Great acting and beautiful cinematography.",
        "Boring and predictable. Don't watch this."
    ]
    
    # Predict sentiments
    for review in test_reviews:
        sentiment = predict_sentiment(review, clf, vectorizer)
        print(f"Review: {review[:50]}...")
        print(f"Sentiment: {sentiment}\n")