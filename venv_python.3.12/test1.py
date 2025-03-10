import nltk

# Confirm the NLTK version
print("NLTK version:", nltk.__version__)

# Ensure the necessary resources are available
nltk.download('punkt')
nltk.download('punkt_tab')

# Test a simple tokenization
from nltk.tokenize import word_tokenize
sample_text = "This is a test sentence. Let's see how it tokenizes!"
tokens = word_tokenize(sample_text)
print("Tokens:", tokens)
