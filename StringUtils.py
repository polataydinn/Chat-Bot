import re
from string import punctuation
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import Constants
import pandas as pd


wordnet_lemma = WordNetLemmatizer()


def wide_contraction(text, contraction_dict):
    contraction_pattern = re.compile('({})'.format('|'.join(contraction_dict.keys())), flags=re.IGNORECASE | re.DOTALL)

    def wide_match(contraction):
        match = contraction.group(0)
        wideContraction = contraction_dict.get(match) \
            if contraction_dict.get(match) \
            else contraction_dict.get(match.lower())
        wideContraction = wideContraction
        return wideContraction

    wideText = contraction_pattern.sub(wide_match, text)
    wideText = re.sub("'", "", wideText)
    return wideText

class StringUtils:

    def main_contraction(self, text):
        text = wide_contraction(text, Constants.contractions_dict)
        return text

    def sayi_kaldir(self, text):
        output = ''.join(c for c in text if not c.isdigit())
        return output

    def nokta_kaldir(self, text):
        return "".join(c for c in text if c not in punctuation)

    def seritle(self, text):
        return " ".join([c for c in text.split() if len(c) > 2])

    def char_kaldir(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
        return text

    def tekrarli_kaldir(self, text):
        text = re.sub("(.)\\1{2,}", "\\1", text)
        return text

    def stopwords_kaldir(self, text):
        stop_words = stopwords.words('english')

        return ' '.join(c for c in nltk.word_tokenize(text) if c not in stop_words)

    def kucuk_karakter(self, text):
        return text.lower()

    def lemma(self, text):
        lemmatize_words = [wordnet_lemma.lemmatize(word) for sent in nltk.sent_tokenize(text) for word in
                           nltk.word_tokenize(sent)]
        return ' '.join(lemmatize_words)

    def setTrimmedQuestions(self, questions):
        questions['prep1'] = questions['Questions'].apply(self.kucuk_karakter)
        questions['prep2'] = questions['prep1'].apply(self.main_contraction)
        questions['prep3'] = questions['prep2'].apply(self.sayi_kaldir)
        questions['prep4'] = questions['prep3'].apply(self.nokta_kaldir)
        questions['prep5'] = questions['prep4'].apply(self.seritle)
        questions['prep6'] = questions['prep5'].apply(self.char_kaldir)
        questions['prep7'] = questions['prep6'].apply(self.tekrarli_kaldir)
        questions['prep8'] = questions['prep7'].apply(self.stopwords_kaldir)
        questions['lemma'] = questions['prep8'].apply(self.lemma)
        return questions

    def setTrimmedAnswers(self, answers):
        answers['prep1'] = answers['Answers'].apply(self.kucuk_karakter)
        answers['prep2'] = answers['prep1'].apply(self.main_contraction)
        answers['prep3'] = answers['prep2'].apply(self.sayi_kaldir)
        answers['prep4'] = answers['prep3'].apply(self.nokta_kaldir)
        answers['prep5'] = answers['prep4'].apply(self.seritle)
        answers['prep6'] = answers['prep5'].apply(self.char_kaldir)
        answers['prep7'] = answers['prep6'].apply(self.tekrarli_kaldir)
        answers['prep8'] = answers['prep7'].apply(self.stopwords_kaldir)
        answers['lemma'] = answers['prep8'].apply(self.lemma)
        return answers

    def setDictionary(self, check):
        check = check.str.extractall('([a-zA_Z]+)')
        check.columns = ['check']
        b = check.reset_index(drop=True)
        check = b['check'].value_counts()

        sozluk = pd.DataFrame({'word': check.index, 'freq': check.values})
        sozluk.index = sozluk['word']
        sozluk.drop('word', axis=1, inplace=True)
        sozluk.sort_values('freq', inplace=True, ascending=False)

        return sozluk

