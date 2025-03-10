import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Download the necessary NLTK data
nltk.download('punkt')

# Initialize the stemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    """
    Tokenizes the text using NLTK and then stems each token.
    Filters out tokens that are not purely alphabetic.
    """
    # Tokenize text into words
    tokens = word_tokenize(text)
    # Keep only alphabetic tokens and convert to lower-case
    tokens = [token.lower() for token in tokens if token.isalpha()]
    # Stem each token
    stems = [stemmer.stem(token) for token in tokens]
    return stems

# --- Data Preparation ---

# Load the filtered CSV file containing the final processed data.
df = pd.read_csv("hotel_reviews_filtered.csv")

# Combine positive and negative review texts into a single column.
df['review_text'] = df['positive_review'].fillna('') + " " + df['negative_review'].fillna('')

# Define a simple heuristic for sentiment based on numeric scores.
def assign_sentiment(row: pd.Series) -> int:
    """
    Assigns sentiment based on the average of 'avg_score' and 'reviewer_score'.
    Reviews with an average score >= 7.5 are labeled as positive (1),
    and those with <= 6.5 are labeled as negative (0).
    """
    try:
        score = (float(row['avg_score']) + float(row['reviewer_score'])) / 2.0
        if score >= 7.5:
            return 1
        elif score <= 6.5:
            return 0
        else:
            return None  # Ambiguous cases are dropped.
    except Exception:
        return None

# Apply the labeling function and drop ambiguous cases.
df['sentiment'] = df.apply(assign_sentiment, axis=1)
df = df.dropna(subset=['sentiment'])
df['sentiment'] = df['sentiment'].astype(int)

# Remove rows with empty review text.
df = df[df['review_text'].str.strip() != '']

# --- Model Building: Text-Only Sentiment Analysis with Stemming ---

# Prepare features and target.
X = df[['review_text']]
y = df['sentiment']

# Split data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a pipeline that uses TfidfVectorizer with our custom tokenizer (that applies stemming)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', tokenizer=tokenize_and_stem,
                                ngram_range=(1, 2), min_df=5)),
    ('clf', LogisticRegression(max_iter=1000, solver='sag', C=0.01))
])

# Train the model.
pipeline.fit(X_train['review_text'], y_train)

# Evaluate the model.
y_pred = pipeline.predict(X_test['review_text'])
print("Classification Report for Text-Only Model with Stemming:")
print(classification_report(y_test, y_pred))

# --- Extract and Plot the Most Influential Words ---

# Get the TfidfVectorizer and LogisticRegression steps from the pipeline.
vectorizer: TfidfVectorizer = pipeline.named_steps['tfidf']
classifier: LogisticRegression = pipeline.named_steps['clf']

# Extract feature names and corresponding coefficients.
feature_names = vectorizer.get_feature_names_out()
coefficients = classifier.coef_[0]

# For binary classification:
# Positive coefficients indicate words associated with positive sentiment,
# and negative coefficients indicate words associated with negative sentiment.
top_n = 20

# Get indices for the top N positive and negative words.
top_positive_indices = np.argsort(coefficients)[-top_n:][::-1]
top_negative_indices = np.argsort(coefficients)[:top_n]

top_positive_words = feature_names[top_positive_indices]
top_positive_values = coefficients[top_positive_indices]

top_negative_words = feature_names[top_negative_indices]
top_negative_values = coefficients[top_negative_indices]

# Plot the top positive words.
plt.figure(figsize=(10, 6))
plt.barh(top_positive_words, top_positive_values, color='green')
plt.xlabel('Coefficient Value')
plt.title('Top Positive Words (with Stemming)')
plt.gca().invert_yaxis()  # Highest values on top
plt.show()

# Plot the top negative words.
plt.figure(figsize=(10, 6))
plt.barh(top_negative_words, top_negative_values, color='red')
plt.xlabel('Coefficient Value')
plt.title('Top Negative Words (with Stemming)')
plt.gca().invert_yaxis()  # Lowest values on top
plt.show()
