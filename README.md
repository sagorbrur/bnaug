# bnaug (Bangla Text Augmentation)
__bnaug__ is a text augmentation tool for Bangla text.

## Installation
```
pip install bnaug
```
- Dependencies
    - pytorch >=1.7.0

## Necessary Model Links
- [word2vec](https://huggingface.co/sagorsarker/bangla_word2vec/resolve/main/bangla_word2vec_gen4.zip)
- [glove vector](https://huggingface.co/sagorsarker/bangla-glove-vectors/resolve/main/bn_glove.300d.zip)

## Sentence Augmentation
### Token Replacement
- Mask generation based augmentation

    ```py
    from bnaug.sentence import TokenReplacement

    tokr = TokenReplacement()
    text = "আমি ঢাকায় বাস করি।"
    output = tokr.masking_based(text, sen_n=5)
    ```

- Word2Vec based augmentation

    ```py
    from bnaug.sentence import TokenReplacement

    tokr = TokenReplacement()
    text = "আমি ঢাকায় বাস করি।"
    model = "msc/bangla_word2vec/bnwiki_word2vec.model"
    output = tokr.word2vec_based(text, model=model, sen_n=5, word_n=5)
    print(output)
    ```

- Glove based augmentation

    ```py
    from bnaug.sentence import TokenReplacement

    tokr = TokenReplacement()
    text = "আমি ঢাকায় বাস করি।"
    vector = "msc/bn_glove.300d.txt"
    output = tokr.glove_based(text, vector_path=vector, sen_n=5, word_n=5)
    print(output)
    ```

### Back Translation
Back translation based augmentation first translate Bangla sentence to English and then again translate the English to Bangla.

```py
from bnaug.sentence import BackTranslation

bt = BackTranslation()
text = "বাংলা ভাষা আন্দোলন তদানীন্তন পূর্ব পাকিস্তানে সংঘটিত একটি সাংস্কৃতিক ও রাজনৈতিক আন্দোলন। "
output = bt.get_augmented_sentences(text)
print(output)

```

### Text Generation
- Paraphrase generation

```py
from bnaug.sentence import TextGeneration

tg = TextGeneration()
text = "বিমানটি যখন মাটিতে নামার জন্য এয়ারপোর্টের কাছাকাছি আসছে, তখন ল্যান্ডিং গিয়ারের খোপের ঢাকনাটি খুলে যায়।"
output = tg.parapharse_generation(text)
print(output)
```

### Random Augmentation
- Random remove part and generate new sentence

    At present it's removing word, stopwords, punctuations, numbers and generate new sentences

    ```py
    from bnaug.sentence import RandomAugmentation

    raug = RandomAugmentation()
    sentence = "আমি ১০০ বাকি দিলাম"
    output = raug.random_remove(sentence)
    print(output)

    ```

    or apply individually

    ```py
    from bnaug import randaug

    text = "১০০ বাকি দিলাম"
    output = randaug.remove_digits(text)
    print(output)

    text = "১০০! বাকি দিলাম?"
    output = randaug.remove_punctuations(text)
    print(output)

    text = "আমি ১০০ বাকি দিলাম"
    randaug.remove_stopwords(text)
    print(output)

    text = "আমি ১০০ বাকি দিলাম"
    randaug.remove_random_word(text)
    print(output)

    text = "আমি ১০০ বাকি দিলাম"
    randaug.remove_random_char(text)
    print(output)
    ```

## Inspired from
- [nlpaug](https://github.com/makcedward/nlpaug)
- [amitness blog post](https://amitness.com/2020/05/data-augmentation-for-nlp/)