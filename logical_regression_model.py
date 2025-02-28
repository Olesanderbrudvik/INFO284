import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from typing import Optional

class SentimentAnalyzer:
    def __init__(self, use_grid_search: bool = False) -> None:
        """
        Initialize the SentimentAnalyzer.
        
        :param use_grid_search: If True, perform grid search to optimize model hyperparameters.
        """
        self.use_grid_search = use_grid_search
        # Build the pipeline during initialization so that it is never None
        self.pipeline: Pipeline = self.build_pipeline()
        self.best_params: Optional[dict] = None

    def build_pipeline(self) -> Pipeline:
        """
        Build the model pipeline with TfidfVectorizer and LogisticRegression.
        
        :return: The constructed pipeline.
        """
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', LogisticRegression(max_iter=1000, solver='saga'))
        ])
        return pipeline

    def run_grid_search(self, X_train, y_train) -> dict:
        """
        Run grid search on the pipeline to optimize hyperparameters.
        
        :param X_train: Training text data.
        :param y_train: Training labels.
        :return: Best parameters found.
        """
        param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__min_df': [5, 10],
            'clf__C': [0.1, 1.0, 10.0]
        }
        grid = GridSearchCV(self.pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        self.pipeline = grid.best_estimator_
        self.best_params = grid.best_params_
        return self.best_params

    def fit(self, X_train, y_train) -> 'SentimentAnalyzer':
        """
        Fit the model. If use_grid_search is True, perform grid search to update the pipeline.
        
        :param X_train: Training text data.
        :param y_train: Training labels.
        :return: The fitted SentimentAnalyzer instance.
        """
        # Rebuild the pipeline to start fresh
        self.pipeline = self.build_pipeline()
        if self.use_grid_search:
            best_params = self.run_grid_search(X_train, y_train)
            print("Best parameters from grid search:", best_params)
        else:
            self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X):
        """
        Predict labels for given text data.
        
        :param X: Text data.
        :return: Predicted labels.
        """
        return self.pipeline.predict(X)

    def score(self, X, y) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        :param X: Test text data.
        :param y: True labels.
        :return: Accuracy score.
        """
        return float(self.pipeline.score(X, y))

if __name__ == '__main__':
    # Load the processed file (only negative_review and positive_review columns are expected)
    df = pd.read_csv("hotel_reviews_filtered.csv")
    df = df[['negative_review', 'positive_review']]
    
    # Create a combined review text by concatenating positive and negative reviews.
    # This gives us one document per review.
    df['review_text'] = df['positive_review'].fillna('') + " " + df['negative_review'].fillna('')
    
    # Define a heuristic to assign sentiment:
    # If the positive review contains as many or more words than the negative review, label as positive (1),
    # otherwise label as negative (0). This is a simple approach to generate labels.
    def assign_sentiment(row):
        pos_words = len(str(row['positive_review']).split())
        neg_words = len(str(row['negative_review']).split())
        return 1 if pos_words >= neg_words else 0

    df['sentiment'] = df.apply(assign_sentiment, axis=1)
    
    # Drop rows with an empty combined review_text
    df = df[df['review_text'].str.strip() != '']
    
    # Split data into training and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        df['review_text'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    # Create an instance of SentimentAnalyzer.
    # Set use_grid_search=True to perform grid search, or False to use default parameters.
    analyzer = SentimentAnalyzer(use_grid_search=False)
    analyzer.fit(X_train, y_train)
    
    # Predict on the test set and print classification metrics
    y_pred = analyzer.predict(X_test)
    print(classification_report(y_test, y_pred))
