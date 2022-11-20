import sys

sys.path.append("../bnaug")
from bnaug import randaug


def test_remove_digits():
    text = "১০০ বাকি দিলাম"
    assert randaug.remove_digits(text) == "বাকি দিলাম"

def test_remove_punctuations():
    text = "১০০! বাকি দিলাম?"
    assert randaug.remove_punctuations(text) == "১০০ বাকি দিলাম"

def test_remove_stopwords():
    text = "আমি ১০০ বাকি দিলাম"
    assert randaug.remove_stopwords(text) == "১০০ বাকি দিলাম"
