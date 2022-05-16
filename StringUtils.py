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


def removeChars(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    return text


def mainContraction(text):
    text = wide_contraction(text, Constants.contractions_dict)
    return text


def removeNumbers(text):
    output = ''.join(c for c in text if not c.isdigit())
    return output


def removeDots(text):
    return "".join(c for c in text if c not in punctuation)


def putLines(text):
    return " ".join([c for c in text.split() if len(c) > 2])


def removeDuplicates(text):
    text = re.sub("(.)\\1{2,}", "\\1", text)
    return text


def removeStopWords(text):
    stop_words = stopwords.words('english')

    return ' '.join(c for c in nltk.word_tokenize(text) if c not in stop_words)


def removeLowerCases(text):
    return text.lower()


def lemma(text):
    lemmatize_words = [wordnet_lemma.lemmatize(word) for sent in nltk.sent_tokenize(text) for word in
                       nltk.word_tokenize(sent)]
    return ' '.join(lemmatize_words)


class StringUtils:
    def setTrimmedQuestions(self, questions):
        questions['prep1'] = questions['Questions'].apply(removeLowerCases)
        questions['prep2'] = questions['prep1'].apply(mainContraction)
        questions['prep3'] = questions['prep2'].apply(removeNumbers)
        questions['prep4'] = questions['prep3'].apply(removeDots)
        questions['prep5'] = questions['prep4'].apply(putLines)
        questions['prep6'] = questions['prep5'].apply(removeChars)
        questions['prep7'] = questions['prep6'].apply(removeDuplicates)
        questions['prep8'] = questions['prep7'].apply(removeStopWords)
        questions['lemma'] = questions['prep8'].apply(lemma)
        return questions

    def setTrimmedAnswers(self, answers):
        answers['prep1'] = answers['Answers'].apply(removeLowerCases)
        answers['prep2'] = answers['prep1'].apply(mainContraction)
        answers['prep3'] = answers['prep2'].apply(removeNumbers)
        answers['prep4'] = answers['prep3'].apply(removeDots)
        answers['prep5'] = answers['prep4'].apply(putLines)
        answers['prep6'] = answers['prep5'].apply(removeChars)
        answers['prep7'] = answers['prep6'].apply(removeDuplicates)
        answers['prep8'] = answers['prep7'].apply(removeStopWords)
        answers['lemma'] = answers['prep8'].apply(lemma)
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

