import random
from bnlp import BasicTokenizer
from bnlp import BengaliWord2Vec, BengaliGlove

from bnlp.corpus import stopwords

from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    pipeline,
)

basic_tokenizer = BasicTokenizer()

MASK_TOKEN = "[MASK]"

class TokenReplacement:
    def masking_based(self, sentence, mask_model=None, sen_n=5, top_k=5):
        if not mask_model:
            model_path = "sagorsarker/bangla-bert-base"
        else:
            model_path = mask_model
        unmasker = pipeline('fill-mask', model=model_path, top_k=top_k)

        tokens = basic_tokenizer.tokenize(sentence)
        output_sentences = []
        for i in range(sen_n):
            try:
                replace_token_index = random.choice(range(len(tokens)))
                replace_token = tokens[replace_token_index]
                rep_text = sentence.replace(replace_token, MASK_TOKEN)
                aug_text = unmasker(rep_text)
                for at in aug_text:
                    seq = at['sequence']
                    output_sentences.append(seq)
            except Exception as e:
                print(e)

        return output_sentences

    def word2vec_based(self, sentence, model, sen_n=5, word_n=5, stopword_remove=False):
        bwv = BengaliWord2Vec()
        tokens = basic_tokenizer.tokenize(sentence)
        if stopword_remove:
            tokens = [x for x in tokens if x not in stopwords]
        output_sentences = []
        for i in range(sen_n):
                replace_token_index = random.choice(range(len(tokens)))
                replace_token = tokens[replace_token_index]
                try:
                    similar_words = bwv.most_similar(model, replace_token)
                    if len(similar_words) > word_n:
                        similar_words = similar_words[:word_n]
                    for sim_word in similar_words:
                        s_word, score = sim_word
                        new_sentence = sentence.replace(replace_token, s_word)
                        output_sentences.append(new_sentence)
                except Exception as e:
                    print(e)
        
        return output_sentences

    def glove_based(self, sentence, vector_path, sen_n=5, word_n=5, stopword_remove=False):
        bnglove = BengaliGlove()
        tokens = basic_tokenizer.tokenize(sentence)
        if stopword_remove:
            tokens = [x for x in tokens if x not in stopwords]
        output_sentences = []
        for i in range(sen_n):
                replace_token_index = random.choice(range(len(tokens)))
                replace_token = tokens[replace_token_index]
                try:
                    similar_words = bnglove.closest_word(vector_path, replace_token)
                    if len(similar_words) > word_n:
                        similar_words = similar_words[:word_n]
                    for sim_word in similar_words:
                        new_sentence = sentence.replace(replace_token, sim_word)
                        output_sentences.append(new_sentence)
                except Exception as e:
                    print(e)
        
        return output_sentences


class BackTranslation:
    pass

class TextGeneration:
    pass
