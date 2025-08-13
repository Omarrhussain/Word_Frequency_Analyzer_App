import re
import collections
import matplotlib.pyplot as plt

class WordAnalyzer:
    def __init__(self):
        self.word_frequencies = {}
        self.word_pattern = re.compile(r'[a-z]+') 
        
    def analyze(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read().lower()
            # Remove punctuation and split into words
            words = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text).split()
            self.word_frequencies = collections.Counter(words)
    
    def get_top_words(self, limit=10):
        return dict(sorted(self.word_frequencies.items(), 
                         key=lambda x: x[1], 
                         reverse=True)[:limit])
    
    def create_chart(self, output_path, limit=10):
        top_words = self.get_top_words(limit)
        plt.figure(figsize=(8, 5))
        plt.bar(top_words.keys(), top_words.values())
        plt.xticks(rotation=45)
        plt.title('Word Frequency Distribution')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()