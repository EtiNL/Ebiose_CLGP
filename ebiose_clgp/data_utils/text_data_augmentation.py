import random
from nltk.corpus import wordnet
from transformers import MarianMTModel, MarianTokenizer
import nltk

nltk.download('wordnet')

# Synonym Replacement
def synonym_replacement(sentence, n=3):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

# Back Translation
def back_translation(sentence, src_lang="en", tgt_lang="fr"):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translated = model.generate(**tokenizer(sentence, return_tensors="pt", padding=True))
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    model_name_back = f'Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}'
    tokenizer_back = MarianTokenizer.from_pretrained(model_name_back)
    model_back = MarianMTModel.from_pretrained(model_name_back)

    translated_back = model_back.generate(**tokenizer_back(translated_text, return_tensors="pt", padding=True))
    translated_back_text = tokenizer_back.decode(translated_back[0], skip_special_tokens=True)

    return translated_back_text
