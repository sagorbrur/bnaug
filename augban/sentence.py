import random
from bnlp import BasicTokenizer

from bnlp.corpus import stopwords
from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    pipeline,
)

basic_tokenizer = BasicTokenizer()

MASK_TOKEN = "[MASK]"

class TokenReplacement:
    def masking_based(self, text, mask_model=None, sen_n=5, top_k=5):
        if not mask_model:
            model_path = "sagorsarker/bangla-bert-base"
        else:
            model_path = mask_model
        unmasker = pipeline('fill-mask', model=model_path, top_k=top_k)

        tokens = basic_tokenizer.tokenize(text)
        output_texts = []
        for i in range(sen_n):
            replace_token_index = random.choice(range(len(tokens)))
            replace_token = tokens[replace_token_index]
            rep_text = text.replace(replace_token, MASK_TOKEN)
            aug_text = unmasker(rep_text)
            for at in aug_text:
                seq = at['sequence']
                output_texts.append(seq)

        return output_texts

    def word2vec_based(self, text):
        pass
    def tfidf_based(self, text):
        pass

class BackTranslation:
    pass

