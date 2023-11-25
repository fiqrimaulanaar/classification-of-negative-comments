import pandas as pd
import re
from nltk.tokenize import WordPunctTokenizer
import itertools
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


def hapus_baris(text):
    return re.sub('\n', ' ', text)


def hapus_backslash(text):
    a = re.sub(r'\\+', '/', text)
    a = re.sub('/n', '\n', a)
    return a


def hapus_quote(text):
    text = re.sub('\[', ' [', text)
    text = re.sub('\]', '] ', text)
    text = re.sub('\[quote[^ ]*\].*?\[\/quote\]', ' ', text)
    text = re.sub('\[[^ ]*\]', ' ', text)
    text = re.sub('&quot;', ' ', text)
    return text


def hapus_link(text):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)


def hapus_spasi_banyak(text):
    return re.sub('  +', ' ', text)


def tokenize_text(text, punct=False):
    text = WordPunctTokenizer().tokenize(text)
    text = [word for word in text if punct or word.isalnum()]
    text = ' '.join(text)
    text = text.strip()
    return text


slang = pd.read_csv('data/slangword.csv')
slang_dict = dict(zip(slang['original'], slang['translated']))


def transform_slang_words(text):
    word_list = text.split()
    word_list_len = len(word_list)
    transformed_word_list = []
    i = 0
    while i < word_list_len:
        if (i + 1) < word_list_len:
            two_words = ' '.join(word_list[i:i+2])
            if two_words in slang_dict:
                transformed_word_list.append(slang_dict[two_words])
                i += 2
                continue
        transformed_word_list.append(
            slang_dict.get(word_list[i], word_list[i]))
        i += 1
    return ' '.join(transformed_word_list)


def remove_non_alphabet(text):
    output = re.sub('[^a-zA-Z ]+', '', text)
    return output


def remove_twitter_ig_formatting(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'\brt\b', '', text)
    return text


removeFactory = StopWordRemoverFactory()
stopwords = removeFactory.create_stop_word_remover()


def removeStopWords(text):
    stop = stopwords.remove(text)
    return stop


def remove_repeating_characters(text):
    return ''.join(''.join(s)[:1] for _, s in itertools.groupby(text))


def preprocess(text):
    text = text.lower()
    # text = hapus_quote(text)
    text = hapus_backslash(text)
    text = hapus_baris(text)
    text = hapus_quote(text)
    text = hapus_link(text)
    text = remove_twitter_ig_formatting(text)
    text = tokenize_text(text)
    text = remove_repeating_characters(text)
    text = remove_non_alphabet(text)
    text = transform_slang_words(text)
    text = remove_non_alphabet(text)
    text = hapus_spasi_banyak(text)
    text = removeStopWords(text)
    # text = stemming(text)
    text = text.lower()
    return text
