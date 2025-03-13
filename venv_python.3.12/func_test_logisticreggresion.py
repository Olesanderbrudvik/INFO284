import os
import re
import string
import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from typing import Optional, Any, cast

# Ensure necessary NLTK resources are available
nltk.download('vader_lexicon')
nltk.download('punkt')

class SentimentAnalyzer:
    def __init__(self, use_grid_search: bool = False) -> None:
        """
        Initializes the SentimentAnalyzer.
        :param use_grid_search: If True, perform grid search during training.
        """
        self.use_grid_search: bool = use_grid_search
        self.sia: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
        self.stemmer: SnowballStemmer = SnowballStemmer("english")
        self.pipeline: Optional[Pipeline] = None

    def custom_tokenizer(self, text: str) -> list[str]:
        """
        Tokenizes text by lowercasing, removing punctuation, tokenizing, and stemming.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        stems = [self.stemmer.stem(token) for token in tokens if token.isalpha()]
        return stems

    def assign_sentiment(self, row: pd.Series, neg_weight: float = 1.5, threshold: float = 0.05) -> int:
        """
        Assigns sentiment based on a weighted combination of VADER scores from the positive and negative reviews.
    
        The positive and negative texts are processed separately:
        - pos_score: VADER compound score for positive_review.
        - neg_score: VADER compound score for negative_review.
    
        The negative score is multiplied by neg_weight to emphasize its impact.
        If the combined score (final_score) is greater than or equal to the threshold,
        the review is classified as positive (1); otherwise, it is classified as negative (0).

        :param neg_weight: Weight for the negative review component (default is 1.5).
        :param threshold: Threshold for classifying a review as positive (default is 0.05).
        """
        pos_text = row['positive_review']
        neg_text = row['negative_review']
    
        # Ensure pos_text is a string; if not, or if it is NaN, set it to an empty string.
        if pd.isna(pos_text) or not isinstance(pos_text, str):
            pos_text = ""
        
        # Ensure neg_text is a string; if not, or if it is NaN, set it to an empty string.
        if pd.isna(neg_text) or not isinstance(neg_text, str):
            neg_text = ""
    
        pos_score = self.sia.polarity_scores(pos_text)['compound'] if pos_text.strip() != "" else 0.0
        neg_score = self.sia.polarity_scores(neg_text)['compound'] if neg_text.strip() != "" else 0.0
    
        final_score = pos_score + (neg_weight * neg_score)
    
        return 1 if final_score >= threshold else 0



    def build_pipeline(self) -> Pipeline:
        """
        Builds and returns the text classification pipeline.
        Uses a custom tokenizer and a custom stop words list (lowercased and stemmed).
        """
        custom_stop_words = list({self.stemmer.stem(word) for word in ENGLISH_STOP_WORDS if word.isalpha()})
        
        tfidf = TfidfVectorizer(
            stop_words=custom_stop_words,
            tokenizer=self.custom_tokenizer,
            ngram_range=(1, 3),
            min_df=3
        )
        clf = LogisticRegression(
            max_iter=1000,
            solver='saga',
            C=1.0,
            class_weight='balanced',
            random_state=42
        )
        self.pipeline = Pipeline([
            ('tfidf', tfidf),
            ('clf', clf)
        ])
        return self.pipeline

    def run_grid_search(self, X_train: list[str], y_train: list[int]) -> dict:
        """
        Runs grid search to optimize hyperparameters on a smaller grid.
        Returns the best parameters.
        """
        if self.pipeline is None:
            self.build_pipeline()
        assert self.pipeline is not None, "Pipeline must be built before running grid search."

        param_grid = {
            'tfidf__ngram_range': [(1, 2), (1, 3)],
            'clf__C': [0.1, 1.0]
        }
        grid = GridSearchCV(self.pipeline, param_grid, cv=3, n_jobs=-2, verbose=1)
        grid.fit(X_train, y_train)
        self.pipeline = grid.best_estimator_
        return grid.best_params_

    def fit(self, X_train: list[str], y_train: list[int]) -> None:
        """
        Trains the model; if use_grid_search is True, runs grid search.
        """
        if self.pipeline is None:
            self.build_pipeline()
        if self.use_grid_search:
            best_params = self.run_grid_search(X_train, y_train)
            print("Best parameters from grid search:", best_params)
        else:
            assert self.pipeline is not None, "Pipeline must be built before fitting."
            self.pipeline.fit(X_train, y_train)

    def predict(self, X_test: list[str]) -> np.ndarray:
        """
        Predicts labels for the provided test set.
        """
        assert self.pipeline is not None, "Pipeline is not built or fitted."
        return cast(np.ndarray, self.pipeline.predict(X_test))

    def evaluate(self, X_test: list[str], y_test: list[int]) -> str:
        """
        Evaluates the model and prints a classification report.
        """
        y_pred = self.predict(X_test)
        report: str = cast(str, classification_report(y_test, y_pred, output_dict=False))
        print(report)
        return report

    def plot_influential_words(self, top_n: int = 20) -> None:
        """
        Plots the top positive and negative words based on the classifier's coefficients.
        """
        assert self.pipeline is not None, "Pipeline must be built and fitted to plot influential words."
        vectorizer: TfidfVectorizer = self.pipeline.named_steps['tfidf']
        classifier: LogisticRegression = self.pipeline.named_steps['clf']
        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[0]

        top_positive_indices = np.argsort(coefficients)[-top_n:][::-1]
        top_negative_indices = np.argsort(coefficients)[:top_n]

        top_positive_words = feature_names[top_positive_indices]
        top_positive_values = coefficients[top_positive_indices]
        top_negative_words = feature_names[top_negative_indices]
        top_negative_values = coefficients[top_negative_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(top_positive_words, top_positive_values, color='green')
        plt.xlabel('Coefficient Value')
        plt.title('Top Positive Words (Improved)')
        plt.gca().invert_yaxis()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.barh(top_negative_words, top_negative_values, color='red')
        plt.xlabel('Coefficient Value')
        plt.title('Top Negative Words (Improved)')
        plt.gca().invert_yaxis()
        plt.show()

    def plot_confusion_matrix(self, X_test: list[str], y_test: list[int]) -> None:
        """
        Computes and plots the confusion matrix for the test set.
        """
        from sklearn.metrics import confusion_matrix
        
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Predicted Negative", "Predicted Positive"],
                    yticklabels=["Actual Negative", "Actual Positive"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()


def main() -> None:
    # Load the processed data file.
    data_file = "hotel_reviews_filtered.csv"
    df = pd.read_csv(data_file)

    # Create combined review text if not already present.
    df['review_text'] = df['positive_review'].fillna('') + " " + df['negative_review'].fillna('')

    # Initialize the SentimentAnalyzer (set grid search flag as desired).
    use_grid_search = True  # Set to True to perform grid search; False to skip it.
    analyzer = SentimentAnalyzer(use_grid_search=use_grid_search)

    # Apply VADER-based sentiment labeling.
    df['sentiment'] = df.apply(analyzer.assign_sentiment, axis=1)
    print("Sentiment distribution:")
    print(df['sentiment'].value_counts())

    # Use only the combined review text for the model.
    X = df['review_text']
    y = df['sentiment']

    # Convert the Series to lists.
    X_list: list[str] = X.tolist()
    y_list: list[int] = y.tolist()

    # Split data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X_list, y_list, test_size=0.2, random_state=42)

    # Build and train the pipeline.
    analyzer.build_pipeline()
    analyzer.fit(X_train, y_train)

    # Evaluate the model.
    analyzer.evaluate(X_test, y_test)
    
    # Plot the confusion matrix.
    analyzer.plot_confusion_matrix(X_test, y_test)

    # Plot the most influential words.
    analyzer.plot_influential_words()


if __name__ == '__main__':
    main()