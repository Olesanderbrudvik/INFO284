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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from typing import Optional, Any, cast

# Sørg for at nødvendige NLTK-ressurser er tilgjengelige
nltk.download('vader_lexicon')
nltk.download('punkt')

class SentimentAnalyzerNB:
    def __init__(self, use_grid_search: bool = False) -> None:
        """
        Initialiserer SentimentAnalyzerNB.
        :param use_grid_search: Hvis True, kjøres grid search under treningen.
        """
        self.use_grid_search: bool = use_grid_search
        self.sia: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
        self.stemmer: SnowballStemmer = SnowballStemmer("english")
        self.pipeline: Optional[Pipeline] = None

    def custom_tokenizer(self, text: str) -> list[str]:
        """
        Tokeniserer tekst ved å gjøre alt små bokstaver, fjerne tegnsetting, tokenisere og stemme.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        stems = [self.stemmer.stem(token) for token in tokens if token.isalpha()]
        return stems

    def assign_sentiment(self, row: pd.Series, neg_weight: float = 1.0, threshold: float = 0.05) -> int:
        """
        Tildeler sentiment basert på en vektet kombinasjon av VADER-skårer for positive og negative anmeldelser.
        """
        pos_text = row['positive_review']
        neg_text = row['negative_review']
    
        if pd.isna(pos_text) or not isinstance(pos_text, str):
            pos_text = ""
        if pd.isna(neg_text) or not isinstance(neg_text, str):
            neg_text = ""
    
        pos_score = self.sia.polarity_scores(pos_text)['compound'] if pos_text.strip() != "" else 0.0
        neg_score = self.sia.polarity_scores(neg_text)['compound'] if neg_text.strip() != "" else 0.0
    
        final_score = pos_score + (neg_weight * neg_score)
    
        return 1 if final_score >= threshold else 0

    def build_pipeline(self) -> Pipeline:
        """
        Bygger og returnerer tekstklassifiserings-pipelinen.
        """
    # Lag et tilpasset stoppordsett basert på stemte versjoner av ENGLISH_STOP_WORDS
        custom_stop_words = list({self.stemmer.stem(word) for word in ENGLISH_STOP_WORDS if word.isalpha()})
    
        tfidf = TfidfVectorizer(
            stop_words=custom_stop_words,
            tokenizer=self.custom_tokenizer,
            ngram_range=(1, 3),
            min_df=3
        )
    
        # Basert på telleverdiene:
        #   Negative (klasse 0): 94068
        #   Positive (klasse 1): 417031
        # Beregning:
        #   ratio = 417031 / 94068 ≈ 4.434
        #   p_neg = 4.434 / (4.434 + 1) ≈ 0.8158
        #   p_pos = 1 / (4.434 + 1) ≈ 0.1842
        # Setter de manuelle priorene for MultinomialNB
        clf = MultinomialNB(alpha=5.0, fit_prior=True, class_prior=[0.8158, 0.1842])
    
        self.pipeline = Pipeline([
            ('tfidf', tfidf),
            ('clf', clf)
        ])
        return self.pipeline


    def run_grid_search(self, X_train: list[str], y_train: list[int]) -> dict:
        """
        Kjører grid search for å optimalisere hyperparametere.
        """
        if self.pipeline is None:
            self.build_pipeline()
        assert self.pipeline is not None, "Pipelinen må bygges før grid search kjøres."

        param_grid = {
            'tfidf__ngram_range': [(1, 2), (1, 3)],
            'clf__alpha': [0.1, 1.0, 5.0]
        }
        grid = GridSearchCV(self.pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        self.pipeline = grid.best_estimator_
        return grid.best_params_

    def fit(self, X_train: list[str], y_train: list[int]) -> None:
        """
        Trener modellen; hvis use_grid_search er True, kjøres grid search.
        """
        if self.pipeline is None:
            self.build_pipeline()
        if self.use_grid_search:
            best_params = self.run_grid_search(X_train, y_train)
            print("Beste parametre fra grid search:", best_params)
        else:
            assert self.pipeline is not None, "Pipelinen må bygges før fitting."
            self.pipeline.fit(X_train, y_train)

    def predict(self, X_test: list[str]) -> np.ndarray:
        """
        Predikerer etiketter for gitt testsett.
        """
        assert self.pipeline is not None, "Pipelinen er ikke bygget eller trent."
        return cast(np.ndarray, self.pipeline.predict(X_test))

    def evaluate(self, X_test: list[str], y_test: list[int]) -> str:
        """
        Evaluerer modellen og skriver ut en klassifiseringsrapport for testsettet.
        """
        y_pred = self.predict(X_test)
        report: str = cast(str, classification_report(y_test, y_pred, output_dict=False))
        print("Classification Report for Test Data:")
        print(report)
        return report

    def evaluate_overfitting(self, X_train: list[str], y_train: list[int],
                             X_test: list[str], y_test: list[int]) -> None:
        """
        Sammenligner modellens ytelse på treningssettet og testsettet for å evaluere overfitting.
        """
        # Evaluer på treningsdata
        y_train_pred = self.predict(X_train)
        train_accuracy = np.mean(np.array(y_train) == y_train_pred)
        train_report = classification_report(y_train, y_train_pred)
        
        # Evaluer på testdata
        y_test_pred = self.predict(X_test)
        test_accuracy = np.mean(np.array(y_test) == y_test_pred)
        test_report = classification_report(y_test, y_test_pred)
        
        print("Overfitting Evaluation:")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report for Training Data:")
        print(train_report)
        print("Classification Report for Test Data:")
        print(test_report)
        
        if train_accuracy - test_accuracy > 0.05:
            print("Advarsel: Modellen kan være overfitted.")
        else:
            print("Modellen ser ikke ut til å overfitte signifikant.")

    def plot_influential_words(self, top_n: int = 20) -> None:
        """
        Plotter de topp positive og negative ordene basert på klassifisererens koeffisienter.
        Merk: For MultinomialNB kan koeffisenter tolkes noe annerledes enn for logistisk regresjon.
        """
        assert self.pipeline is not None, "Pipelinen må bygges og trenes for å plotte påvirkende ord."
        vectorizer = self.pipeline.named_steps['tfidf']
        # MultinomialNB har ikke koeffisienter på samme måte, men vi kan se på log-sannsynligheter:
        clf: MultinomialNB = self.pipeline.named_steps['clf']
        if not hasattr(clf, "feature_log_prob_"):
            print("Modellen har ikke attributtet 'feature_log_prob_'.")
            return
        
        feature_names = vectorizer.get_feature_names_out()
        # Differansen mellom log-sannsynligheter for de to klassene:
        coef_diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        
        top_positive_indices = np.argsort(coef_diff)[-top_n:][::-1]
        top_negative_indices = np.argsort(coef_diff)[:top_n]

        top_positive_words = feature_names[top_positive_indices]
        top_positive_values = coef_diff[top_positive_indices]
        top_negative_words = feature_names[top_negative_indices]
        top_negative_values = coef_diff[top_negative_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(top_positive_words, top_positive_values)
        plt.xlabel('Log sannsynlighetsdifferanse')
        plt.title('Topp positive ord')
        plt.gca().invert_yaxis()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.barh(top_negative_words, top_negative_values)
        plt.xlabel('Log sannsynlighetsdifferanse')
        plt.title('Topp negative ord')
        plt.gca().invert_yaxis()
        plt.show()

    def plot_confusion_matrix(self, X_test: list[str], y_test: list[int]) -> None:
        """
        Beregner og plotter forvirringsmatrisen for testsettet.
        """
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
    # Last inn data
    data_file = "hotel_reviews_filtered.csv"
    df = pd.read_csv(data_file)

    # Kombiner tekst fra positive og negative anmeldelser
    df['review_text'] = df['positive_review'].fillna('') + " " + df['negative_review'].fillna('')

    # Initialiser SentimentAnalyzerNB (sett grid search-flagget etter ønske)
    use_grid_search = False  # Sett til True for å kjøre grid search
    analyzer = SentimentAnalyzerNB(use_grid_search=use_grid_search)

    # Bruk VADER-basert sentiment tildeling
    df['sentiment'] = df.apply(analyzer.assign_sentiment, axis=1)
    print("Sentiment distribusjon:")
    print(df['sentiment'].value_counts())

    # Bruk kun kombinert review tekst for modellen
    X = df['review_text']
    y = df['sentiment']

    # Konverter Series til lister
    X_list: list[str] = X.tolist()
    y_list: list[int] = y.tolist()

    # Splitt data i trenings- og testsett
    X_train, X_test, y_train, y_test = train_test_split(X_list, y_list, test_size=0.2, random_state=42)

    # Bygg og tren pipelinen
    analyzer.build_pipeline()
    analyzer.fit(X_train, y_train)

    # Evaluer modellen på testdata
    analyzer.evaluate(X_test, y_test)
    
    # Evaluer overfitting ved å sammenligne trenings- og testytelse
    analyzer.evaluate_overfitting(X_train, y_train, X_test, y_test)

    # Plot forvirringsmatrisen
    analyzer.plot_confusion_matrix(X_test, y_test)

    # Plot de mest påvirkende ordene
    analyzer.plot_influential_words()

if __name__ == '__main__':
    main()
