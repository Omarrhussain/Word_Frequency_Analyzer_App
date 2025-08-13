import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class NLTKWordAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def analyze(self, filepath, top_n=10):
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read().lower()
            words = word_tokenize(text)
            filtered_words = [word for word in words if word.isalnum() and word not in self.stop_words]
            word_freq = Counter(filtered_words)
            
        return word_freq.most_common(top_n)