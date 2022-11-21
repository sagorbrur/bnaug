from bnaug.sentence import TokenReplacement, TextGeneration

def test_mask_based():
    tokr = TokenReplacement()
    text = "আমি ঢাকায় বাস করি।"
    output = tokr.masking_based(text, mask_model='csebuetnlp/banglabert_generator')
    print(output)

def test_word2vec_based():
    tokr = TokenReplacement()
    text = "আমি ঢাকায় বাস করি।"
    model = "msc/bangla_word2vec/bnwiki_word2vec.model"
    output = tokr.word2vec_based(text, model=model)
    print(output)

def test_glove_based():
    tokr = TokenReplacement()
    text = "আমি ঢাকায় বাস করি।"
    model = "msc/bn_glove.300d.txt"
    output = tokr.glove_based(text, vector_path=model)
    print(output)

def test_backtranslation():
    from bnaug.sentence import BackTranslation

    bt = BackTranslation()
    text = "বাংলা ভাষা আন্দোলন তদানীন্তন পূর্ব পাকিস্তানে সংঘটিত একটি সাংস্কৃতিক ও রাজনৈতিক আন্দোলন। "
    output = bt.get_augmented_sentences(text)
    print(output)

def test_paraphrase():
    from bnaug.sentence import TextGeneration

    tg = TextGeneration()
    text = "বিমানটি যখন মাটিতে নামার জন্য এয়ারপোর্টের কাছাকাছি আসছে, তখন ল্যান্ডিং গিয়ারের খোপের ঢাকনাটি খুলে যায়।"
    output = tg.parapharse_generation(text)
    print(output)
    # প্লেনটা এয়ারপোর্টের কাছে অবতরণ করার সময়, ল্যান্ডিং গিয়ার প্যানেলের ঢাকনাটা খুলে দেওয়া'

def test_random_remove():
    from bnaug.sentence import RandomAugmentation

    ru = RandomAugmentation()
    sentence = "আমি ১০০ বাকি দিলাম"
    output = ru.random_remove(sentence)
    print(output)

if __name__ == "__main__":
    test_mask_based()
    # test_parapharse()
    # test_word2vec_based()
    # test_glove_based()
    # test_backtranslation()
    # test_paraphrase()
    # test_random_remove() # ['আমি  বাকি দিলাম', 'আমি ১০০ বাকি দিলাম', '১০০ বাকি দিলাম', 'আমি ১০০ বাকি']

# ['আমি এখানে বাস করি ।', 'আমি সেখানে বাস করি ।', 'আমি বাস করি ।', 'আমি বাংলাদেশে বাস করি ।', 
# 'আমি ওখানে বাস করি ।', 'আমি ঢাকায বাস করি ।', 'আমি ঢাকায ভ্রমণ করি ।', 'আমি ঢাকায কাজ করি ।', 
# 'আমি ঢাকায ব্যবহার করি ।', 'আমি ঢাকায পালন করি ।', 'আমরা ঢাকায বাস করি ।', 'আমি ঢাকায বাস করি ।', 
# 'এখানে ঢাকায বাস করি ।', 'সেখানে ঢাকায বাস করি ।', 'বাসে ঢাকায বাস করি ।', 'আমি ঢাকায বাস করি ।', 
# 'আমি ঢাকায বাস করছি ।', 'আমি ঢাকায বাস করেছি ।', 'আমি ঢাকায বাস করতাম ।', 'আমি ঢাকায বাস করিনি ।', 
# 'আমরা ঢাকায বাস করি ।', 'আমি ঢাকায বাস করি ।', 'এখানে ঢাকায বাস করি ।', 'সেখানে ঢাকায বাস করি ।', 
# 'বাসে ঢাকায বাস করি ।']

# ['আমি ঢাকায় বাস করিএবং', 'আমি ঢাকায় বাস করি.', 'আমি ঢাকায় বাস করিযাতে', 'আমি ঢাকায় বাস করিযা', 
# 'আমি ঢাকায় বাস করিযেখানে', 'আমি ঢাকায় বসবাস করি।', 'আমি ঢাকায় যাতায়াত করি।', 'আমি ঢাকায় চলাচল করি।', 
# 'আমি ঢাকায় ঘোরাফেরা করি।', 'আমি ঢাকায় ভ্রমণ করি।', 'আমি ঢাকায় বাস করছি।', 'আমি ঢাকায় বাস করব।', 
# 'আমি ঢাকায় বাস করেছি।', 'আমি ঢাকায় বাস করুন।', 'আমি ঢাকায় বাস করো।', 'আমি ঢাকায় বসবাস করি।', 
# 'আমি ঢাকায় যাতায়াত করি।', 'আমি ঢাকায় চলাচল করি।', 'আমি ঢাকায় ঘোরাফেরা করি।', 'আমি ঢাকায় ভ্রমণ করি।', 
# 'আমি ঢাকায় বসবাস করি।', 'আমি ঢাকায় যাতায়াত করি।', 'আমি ঢাকায় চলাচল করি।', 'আমি ঢাকায় ঘোরাফেরা করি।', 
# 'আমি ঢাকায় ভ্রমণ করি।']

# ['আমি ঢাকায় বাস করি।', 'আমার ঢাকায় বাস করি।', 'করি ঢাকায় বাস করি।', 'আমাকে ঢাকায় বাস করি।', 'আমরা ঢাকায় বাস করি।', 
# 'আমি ঢাকায় বাস করি।', 'আমি ঢাকায় বসবাস করি।', 'আমি ঢাকায় মিনিবাস করি।', 'আমি ঢাকায় ভলভো করি।', 
# 'আমি ঢাকায় লোকের করি।', 'আমি ঢাকায় বাস করি।', 'আমি ঢাকায় বসবাস করি।', 'আমি ঢাকায় মিনিবাস করি।', 
# 'আমি ঢাকায় ভলভো করি।', 'আমি ঢাকায় লোকের করি।']