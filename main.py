import nltk
import pandas as pd
from nltk.corpus import stopwords
import warnings
from StringUtils import StringUtils
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

# cosinus similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Questions cosinus similarity
vectorizer = CountVectorizer().fit_transform(dataset["Questions"])
vectors = vectorizer.toarray()


def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)

    return cosine_similarity(vec1, vec2)[0][0]

print("Questions Cosine similarity:{}".format(cosine_sim_vectors(vectors[0], vectors[1])))

text = dataset['Questions']
tfidf = TfidfVectorizer(use_idf=True, analyzer='word', stop_words='english', token_pattern=r'\b[^\d\W]+\b', ngram_range=(1,2))
X_train = tfidf.fit_transform(text)
lsvc = LinearSVC(random_state = 2021)
lsvc.fit(X_train, y)
vectorizer = CountVectorizer().fit_transform(answers["Answers"])
vectors = vectorizer.toarray()

print("Answers Cosine similarity:{}".format(cosine_sim_vectors(vectors[0], vectors[1])))

i = 0
while i == 0:
    question = input("You : ")
    search_test = [
        question
    ]
    search_engine = tfidf.transform(search_test)
    result = lsvc.predict(search_engine)
    faq_data = dataset.loc[dataset.isin([result]).any(axis=1)]
    for question in result:
        faq_data = dataset.loc[dataset.isin([question]).any(axis=1)]
        if faq_data['Answers'].values == ['What you tell yourself about a situation affects how you feel and what you do. Sometimes your interpretation of a situation can get distorted and you only focus on the negative aspects—this is normal and expected. However, when you interpret situations too negatively, you might feel worse. You\'re also more likely to respond to the situation in ways that are unhelpful in the long term. \n These automatic thoughts and assumptions are sometimes called thinking traps. Everyone falls into unbalanced thinking traps from time to time. You\'re most likely to distort your interpretation of things when you already feel sad, angry, anxious, depressed or stressed. You\'re also more vulnerable to thinking traps when you\'re not taking good care of yourself, like when you\'re not eating or sleeping well. \n Here are some common thinking traps: \n Thinking that a negative situation is part of a constant cycle of bad things that happen. People who overgeneralize often use words like "always" or "never." \n I was really looking forward to that concert, and now it’s cancelled. This always happens to me! I never get to do fun things! \n Seeing things as only right or wrong, good or bad, perfect or terrible. People who think in black and white terms see a small mistake as a total failure. \n I wanted to eat healthier, but I ate too many snacks today. This plan is a total failure! \n Saying only negative things about yourself or other people. \n I made a mistake. I\'m stupid! My boss told me that I made a mistake. My boss is a total jerk! \n Predicting that something bad will happen without any evidence. \n I\'ve been doing what I can to stay home and reduce the risks, but I just know that I\'m going to get sick. \n Focusing only on the negative parts of a situation and ignoring anything good or positive. \n I know there\'s a lot I can do at home, but I\'m just so sick of this. Everything is terrible. \n Believing that bad feelings or emotions reflect the situation. \n I feel scared and overwhelmed right now, so that must mean everything is very bad and will never get better. \n Telling yourself how you "should" or "must" act. \n I should be able to handle this without getting upset and crying! \n Here are helpful strategies to challenge common thinking traps. Many people find their mood and confidence improve after working through these skills. You can also find worksheets to help you go through each step at www.heretohelp.bc.ca. \n Don\'t try to get out of a thinking trap by just telling yourself to stop thinking that way. \n This doesn\'t let you look at the evidence and challenge the thinking trap. When you try and push upsetting thoughts away, they are more likely to keep popping back into your mind. \n Ask yourself the following questions when something upsetting happens: \n What is the situation? What actually happened? Only include facts that everyone would agree on. \n What are your thoughts? What are you telling yourself? \n What are your emotions? How do you feel? \n What are your behaviours? How are you reacting? What are you doing to cope? \n Take a look at the thoughts you\'ve listed. Are you using any of the thinking traps and falling into distorted thinking patterns? It\'s common to fall into more than one thinking trap. Go back to the thinking trap list and identify which ones apply to you and your current situation. \n The best way to break a thinking trap is to look at your thoughts like a scientist and consider the hard facts. Use the evidence you\'ve collected to challenge your thinking traps. Here are some ways to do that: \n Try to find evidence against the thought. If you make a mistake at work, you might automatically think, "I can\'t do anything right! I must be a terrible employee!" When this thought comes up, you might challenge it by asking, "Is there any evidence to support this thought? Is there any evidence to disprove this thought?" You might quickly realize that your boss has complimented your work recently, which doesn\'t support the idea that you\'re a bad employee. \n Ask yourself, "Would I judge other people if they did the same thing? Am I being harder on myself than I am on other people?" This is a great method for challenging thinking traps that involve harsh self-criticism. \n Find out whether other people you trust agree with your thoughts. For example, you might have trouble with one of your kids and think, "Good parents wouldn\'t have this kind of problem." To challenge this thought, you can ask other parents if they\'ve ever had any problems with their kids. \n Test your beliefs in person. For example, if you think that your friends don\'t care about you, call a few friends and make plans to start a regular video call. If you assumed that they will all say no, you may be pleasantly surprised to hear that they do want to see you. \n Once you have worked through some challenges, try to think of a more balanced thought to replace the old thinking traps. Let\'s use the following example: \n I feel sad and overwhelmed. I\'m having a hard time figuring out what to do. \n I\'m the worst! I should be able to handle this! \n Labeling \n \'Should\' statements \n Examine the evidence: I have a lot of challenges right now. I\'m worried about my family and everything seems to change so quickly. I\'ve successfully handled complicated situations in the past, so I know I can do this. \n It\'s okay to feel upset right now—there\'s a lot going on. I\'m going to think about how I got through past situations and see what worked for me. I\'m trying to do a lot on my own, so I\'m going to talk to my family so we can make a plan and work together. \n Try the Healthy Thinking Worksheet at www.heretohelp.bc.ca \n Check out Anxiety Canada\'s articles Helpful Thinking and Thinking Traps \n This page is adapted from Wellness Module 8: Healthy Thinking at www.heretohelp.bc.ca/wellness-module/wellness-module-8-healthy-thinking.']:
            print("Please enter a valid question!")
        else:
            print("Bot: ", faq_data['Answers'].values)
