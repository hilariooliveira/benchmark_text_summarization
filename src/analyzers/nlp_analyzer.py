import nltk

from nltk.corpus import stopwords
from string import punctuation
from src.basic_classes.classes import Document, Sentence, Token
from src.utils import utils


class NLPAnalyzer:
    def __init__(self):
        self.punctuation_symbols = list(punctuation)
        self.punctuation_symbols.append("''")
        self.punctuation_symbols.append("``")
        self.stopwords = set(stopwords.words('english') + self.punctuation_symbols)
        self.stemmer = nltk.stem.RSLPStemmer()

    def process_document(self, document: Document):
        sentences_nltk = nltk.sent_tokenize(document.text)
        for id_sent, sentence_nltk in enumerate(sentences_nltk):
            tokens_nltk = nltk.word_tokenize(sentence_nltk.lower())
            sentence = Sentence()
            sentence.id = id_sent + 1
            sentence.text = sentence_nltk
            id_token = 1
            for token_nltk in tokens_nltk:
                is_stop_word = False
                if token_nltk in self.stopwords or token_nltk in self.punctuation_symbols:
                    is_stop_word = True
                stem = self.stemmer.stem(token_nltk)
                token = Token()
                token.id = id_token
                token.text = token_nltk
                token.stem = stem
                token.is_stop_word = is_stop_word
                token.type = utils.get_token_type(token.text, self.punctuation_symbols)
                sentence.add_token(token)
                if not token.is_stop_word:
                    sentence.add_token_no_stopwords(token)
                id_token += 1
            document.add_sentence(sentence)
