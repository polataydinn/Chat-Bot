import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
import warnings
from CosineSimilarity import CosineSimilarity
from StringUtils import StringUtils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.metrics import cosine_similarity


def downloadNltkFiles():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    stopwords.words('english')


downloadNltkFiles()
warnings.filterwarnings("ignore")

dataset = pd.read_csv('Mental_Health_FAQ.csv')
print(dataset.head())

questions = dataset[['Question_ID', 'Questions']]
answers = dataset[['Question_ID', 'Answers']]

stringUtils = StringUtils()

questions = stringUtils.setTrimmedQuestions(questions)
answers = stringUtils.setTrimmedAnswers(answers)

clearDictionary = stringUtils.setDictionary(questions['lemma'])
clearDictionary[:30].plot(kind='barh', figsize=(10, 10))

label = LabelEncoder()
dataset["AnswerEncode"] = label.fit_transform(dataset["Answers"])
X = dataset["Questions"]
y = dataset["AnswerEncode"].values
countVectorizer = TfidfVectorizer()
X = countVectorizer.fit_transform(X)
countVectorizer.get_feature_names()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
y_pred = lsvc.predict(X_test)

print(mean_squared_error(y_test, y_pred))
print("Model Score: {}".format(lsvc.score(X_train, y_train)))

countVectorizer = CountVectorizer().fit_transform(questions["Questions"])
vectors = countVectorizer.toarray()
cosineSimilarity = CosineSimilarity()

def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]


print("Questions Cosine similarity:{}".format(cosine_sim_vectors(vectors[0], vectors[1])))
countVectorizer = CountVectorizer().fit_transform(answers["Answers"])
vectors = countVectorizer.toarray()
print("Answers Cosine similarity:{}".format(cosine_sim_vectors(vectors[0], vectors[1])))

i = 0
while i == 0:
    question = input("You : ")
    search_test = [
        question
    ]
    print(search_test)
    search_engine = tfidf.transform(search_test)
    result = lsvc.predict(search_engine)
    faq_data = dataset.loc[dataset.isin([result]).any(axis=1)]
    for question in result:
        faq_data = dataset.loc[dataset.isin([question]).any(axis=1)]
        print("Bot: ", faq_data['Answers'].values)
