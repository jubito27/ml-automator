import nltk
from nltk.corpus import stopwords
import string
import re
from nltk.stem.porter import PorterStemmer

# ensure stopwords resource is available
nltk.download('stopwords', quiet=True)

# Prepare stopwords set once for efficiency
_stop_words = set(stopwords.words('english'))

def transform(text):
    # normalize and ensure string
    text = '' if text is None else str(text)
    text = text.lower()

    # simple tokenization using regex (keeps alphanumeric tokens)
    tokens = re.findall(r"\w+", text)
    tokens = [t for t in tokens if t.isalnum()]

    # filter out stopwords and punctuation
    result_tokens = [t for t in tokens if t not in _stop_words and t not in string.punctuation]
    ps = PorterStemmer()
    result = []
    for i in result_tokens:
        result.append(ps.stem(i))

    return " ".join(result)

