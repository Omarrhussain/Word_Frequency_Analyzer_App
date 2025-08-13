import os, tempfile
import pytest
from analyzer import WordAnalyzer
from nltk_analyzer import NLTKWordAnalyzer  # Assuming NLTKWordAnalyzer is defined in nltk_analyzer.py

@pytest.fixture
def small_textfile(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello world! Hello Python. Python is great.", encoding='utf-8')
    return file_path
    
@pytest.fixture 
def large_textfile(tmp_path):
    file_path = tmp_path / "large_sample.txt"
    words = ["python", "flask", "data", "analyze", "word", "test"]
    file_path.write_text((" ".join(words) + " ") * 50000, encoding="utf-8")
    return file_path

@pytest.fixture 
def sample_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello world! This is a test file. Hello again.", encoding='utf-8')
    return file_path

def test_word_count(small_textfile):
    analyzer = WordAnalyzer()
    analyzer.analyze(small_textfile)
    freqs = analyzer.word_frequencies
    assert freqs["hello"] == 2
    assert freqs["python"] == 2
    assert freqs["world"] == 1
    assert len(analyzer.word_frequencies) > 0
    
def test_top_words_order(small_textfile):
    analyzer = WordAnalyzer()
    analyzer.analyze(small_textfile)
    top_words = list(analyzer.get_top_words(2).keys())
    assert top_words[0] == "hello" or top_words[0] == "python"
    assert len(top_words) == 2
    
def test_large_file_handling(large_textfile):
    analyzer = WordAnalyzer()
    analyzer.analyze(large_textfile)
    top_words = analyzer.get_top_words(3)
    assert isinstance(top_words, dict)
    assert len(top_words) == 3
    assert(all(isinstance(word, str) for word in top_words.keys()))
    assert(all(isinstance(count, int) for count in top_words.values()))
    
def test_nltk_word_count(sample_file):  
    analyzer = NLTKWordAnalyzer()
    result = analyzer.analyze(sample_file, top_n=5) 
    assert isinstance(result, list)
    assert all(isinstance(item, tuple) and isinstance(item[0], str) and isinstance(item[1], int) 
              for item in result)
    assert len(result) <= 5

def test_nltk_stop_words(tmp_path):
    test_file = tmp_path / "stopwords_test.txt"
    text = "This is a sample text with common stopwords like the and is."
    test_file.write_text(text, encoding='utf-8')
    analyzer = NLTKWordAnalyzer()
    result = analyzer.analyze(test_file, top_n=10)
    words_only = [word for word, _ in result]
    assert "the" not in words_only
    assert "is" not in words_only
    
def test_bar_chart_creation(sample_file):
    analyzer = WordAnalyzer()
    analyzer.analyze(sample_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
        chart_path = tmp_img.name
    analyzer.create_chart(chart_path, 5)
    assert os.path.exists(chart_path)
    # Clean up the chart file after test
    os.remove(chart_path)