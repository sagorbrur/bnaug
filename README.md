# Augban (Bangla Text Augmentation)
Augban is a text augmentation tool for Bangla text.

## Installation
```
pip install augban
```

## Sentence Augmentation
### Token Replacement
- Mask generation based augmentation

    ```py
    from augban.sentence import TokenReplacement

    tokr = TokenReplacement()
    text = "আমি ঢাকায় বাস করি।"
    output = tokr.masking_based(text)
    ```

- Word2Vec based augmentation

    ```py
    from augban.sentence import TokenReplacement

    tokr = TokenReplacement()
    text = "আমি ঢাকায় বাস করি।"
    model = "msc/bangla_word2vec/bnwiki_word2vec.model"
    output = tokr.word2vec_based(text, model=model)
    print(output)
    ```

- Glove based augmentation

    ```py
    from augban.sentence import TokenReplacement

    tokr = TokenReplacement()
    text = "আমি ঢাকায় বাস করি।"
    vector = "msc/bn_glove.300d.txt"
    output = tokr.glove_based(text, vector_path=vector)
    print(output)
    ```