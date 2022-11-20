import re
import random
from bnlp.corpus import stopwords, punctuations


STOPWORDS = set(stopwords)
PUNCTUATIONS = set(punctuations)

def remove_digits(text):
    return re.sub(r'[০-৯]+', '', text).strip()

def remove_punctuations(text):
    return ''.join([c for c in text if c not in PUNCTUATIONS])

def remove_stopwords(text):
    words = text.split()
    new_text = []
    for word in words:
        if word not in STOPWORDS:
            new_text.append(word)
    return ' '.join(new_text)

def remove_random_word(text):
    words = text.split()
    words.remove(random.choice(words))
    return ' '.join(words)

def remove_random_char(text):
    return ''.join([c for c in text if random.random() < 0.7])
