import pickle
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)


class SentimentService:
    clf = None
    vectorizer = None

    @classmethod
    def load_models(cls):
        if cls.clf is None or cls.vectorizer is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)

            model_path = os.path.join(project_root, "models", "comment_sentiments.pkl")
            vectorizer_path = os.path.join(project_root, "models", "transformed.pkl")
            try:
                with open(model_path, "rb") as f:
                    cls.clf = pickle.load(f)
                with open(vectorizer_path, "rb") as f:
                    cls.vectorizer = pickle.load(f)
                logging.info(f"Models Loaded Successfully!")
            except FileNotFoundError as e:
                logging.error(f"Models Loading Failed: {e}")
                raise e

        return cls.clf, cls.vectorizer

    @classmethod
    def predict(cls, review_text):
        clf, vectorizer = cls.load_models()
        review_vector = vectorizer.transform([review_text])
        prediction = clf.predict(review_vector)[0]
        return "Good" if prediction == 1 else "Bad"


if __name__ == "__main__":
    SentimentService.load_models()
        