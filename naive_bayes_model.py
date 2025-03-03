import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Optional

class SentimentAnalyzer:
    def __init__(self, data_path: str, n_features: int = 10000, sample_frac: Optional[float] = None):
        """
        Initialiserer sentimentanalysatoren.
        
        :param data_path: Filsti til datasettet (CSV-fil).
        :param n_features: Antall funksjoner i HashingVectorizer.
        :param sample_frac: Fraksjon av datasettet som skal brukes (hvis du vil bruke en del av dataene).
        """
        self.data_path = data_path
        self.n_features = n_features
        self.sample_frac = sample_frac
        self.model_pipeline = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        
    def load_and_process_data(self):
        """Laster inn og prosesserer datasettet."""
        df = pd.read_csv(self.data_path)

        # Kombiner positive og negative anmeldelser til én tekst
        df['review_text'] = df['Positive_Review'].fillna('') + " " + df['Negative_Review'].fillna('')
        
        # Fjern tomme anmeldelser
        df = df[df['review_text'].str.strip() != '']
        
        # Opprett sentimentkolonne basert på ordtelling
        df['sentiment'] = df.apply(lambda row: 1 if row['Review_Total_Positive_Word_Counts'] >= row['Review_Total_Negative_Word_Counts'] else 0, axis=1)

        # Reduser datasettet hvis sample_frac er spesifisert
        if self.sample_frac:
            df = df.sample(frac=self.sample_frac, random_state=42)

        # Splitt data i trenings- og testsett
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df['review_text'], df['sentiment'], test_size=0.2, random_state=42
        )

    def build_pipeline(self):
        """Bygger pipeline med HashingVectorizer og Naive Bayes."""
        self.model_pipeline = Pipeline([
            ('hash', HashingVectorizer(stop_words='english', n_features=self.n_features, alternate_sign=False)),
            ('clf', MultinomialNB())
        ])

    def train(self):
        if self.model_pipeline is None:
            self.build_pipeline()
        self.model_pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Evaluerer modellen og skriver ut en rapport."""
        y_pred = self.model_pipeline.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        print(report)
        return report

    def run(self):
        """Kjører hele sentimentanalyseprosessen."""
        self.load_and_process_data()
        self.train()
        return self.evaluate()

# Kjør analysen
if __name__ == "__main__":
    analyzer = SentimentAnalyzer(data_path="hotel_reviews.csv", sample_frac=0.1)
    analyzer.run()