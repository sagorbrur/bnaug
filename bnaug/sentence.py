import random
from bnlp import BasicTokenizer
from bnlp import BengaliWord2Vec, BengaliGlove

from bnlp.corpus import stopwords
from transformers import (
    AutoModelWithLMHead,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

from bnaug import util
from bnaug import randaug

basic_tokenizer = BasicTokenizer()

MASK_TOKEN = "[MASK]"
SPECIAL_TOKEN_LIST = ['<pad>', '</s>']

class TokenReplacement:
    def masking_based(self, sentence, mask_model=None, sen_n=5, top_k=5, device=-1):
        if not mask_model:
            model_path = "sagorsarker/bangla-bert-base"
        else:
            model_path = mask_model
        unmasker = pipeline('fill-mask', model=model_path, top_k=top_k, device=device)

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
    def __init__(self, model_bn_en_path=None, model_en_bn_path=None, top_k=10, top_p=0.5, num_beams=1):
        if not model_bn_en_path:
            model_bn_en_path = "csebuetnlp/banglat5_nmt_bn_en"
        if not model_en_bn_path:
            model_en_bn_path = "csebuetnlp/banglat5_nmt_en_bn"

        self.model_bn2en = AutoModelForSeq2SeqLM.from_pretrained(model_bn_en_path)
        self.tokenizer_bn2en = AutoTokenizer.from_pretrained(model_bn_en_path, use_fast=False)

        self.model_en2bn = AutoModelForSeq2SeqLM.from_pretrained(model_en_bn_path)
        self.tokenizer_en2bn = AutoTokenizer.from_pretrained(model_en_bn_path, use_fast=False)
        self.top_k = top_k
        self.top_p = top_p
        self.num_beams = num_beams

    def get_augmented_sentences(self, sentence):
        bn_inputs = self.tokenizer_bn2en.encode(sentence, return_tensors="pt")
        en_outputs = self.model_bn2en.generate(
            bn_inputs, 
            top_k=self.top_k, 
            top_p=self.top_p, 
            num_beams=self.num_beams
        )
        # print(outputs)
        augment_sentences = []
        for out in en_outputs:
            en_sen = self.tokenizer_bn2en.decode(out)
            en_sen = util.replace_special_token_to_empty(en_sen, SPECIAL_TOKEN_LIST)
            en_sen = en_sen.strip()
            en_inputs = self.tokenizer_en2bn.encode(sentence, return_tensors="pt")
            bn_outputs = self.model_en2bn.generate(
                en_inputs, 
                top_k=self.top_k, 
                top_p=self.top_p, 
                num_beams=self.num_beams
            )
            for out in bn_outputs:
                bn_sen = self.tokenizer_en2bn.decode(out)
                bn_sen = util.replace_special_token_to_empty(bn_sen, SPECIAL_TOKEN_LIST)
                augment_sentences.append(bn_sen)

        return augment_sentences


class TextGeneration:
    def parapharse_generation(self, sentence, model_path=None, top_k=5, top_p=0.6, num_beams=5):
        if not model_path:
            model_path = "csebuetnlp/banglat5_banglaparaphrase"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        inputs = tokenizer.encode(sentence, return_tensors="pt")
        outputs = model.generate(inputs, top_k=top_k, top_p=top_p, num_beams=num_beams)
        augmented_sentences = []
        for out in outputs:
            para_sen = tokenizer.decode(out)
            para_sen = util.replace_special_token_to_empty(para_sen, SPECIAL_TOKEN_LIST)
            augmented_sentences.append(para_sen)
        
        return augmented_sentences

class RandomAugmentation:
    def random_remove(self, sentence, char_remove=False):
        augmented_sentences = []
        augmented_sentences.append(randaug.remove_digits(sentence))
        augmented_sentences.append(randaug.remove_punctuations(sentence))
        augmented_sentences.append(randaug.remove_stopwords(sentence))
        augmented_sentences.append(randaug.remove_random_word(sentence))
        if char_remove:
            augmented_sentences.append(randaug.remove_random_char(sentence))

        return augmented_sentences




