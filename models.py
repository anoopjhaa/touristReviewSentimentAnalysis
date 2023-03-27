from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

X_train, X_test, y_train, y_test, X_train_CountVec, X_train_TFIDF, TFIDF_Vectorizer, Count_Vectorizer = [None] * 8
statistics = []


def getVectorizers():
    return [Count_Vectorizer, TFIDF_Vectorizer]


def getStatistics():
    return statistics


def initializeModels(df):
    global X_train, X_test, y_train, y_test, X_train_CountVec, X_train_TFIDF, TFIDF_Vectorizer, Count_Vectorizer, statistics
    X_train, X_test, y_train, y_test = train_test_split(
        list(df['Review']), list(df['Sentiment']), test_size=0.2, random_state=42)

    Count_Vectorizer = CountVectorizer()
    X_train_CountVec = Count_Vectorizer.fit_transform(X_train)

    TFIDF_Vectorizer = TfidfVectorizer()
    X_train_TFIDF = TFIDF_Vectorizer.fit_transform(X_train)

    statistics.append(CountVec_NaiveBayes(df))
    statistics.append(CountVec_SVM(df))
    # statistics.append(CountVec_RF(df))
    statistics.append(TFIDF_NaiveBayes(df))
    statistics.append(TFIDF_SVM(df))
    # statistics.append(TFIDF_RF(df))


def CountVec_NaiveBayes(df):
    nb = MultinomialNB()
    nb.fit(X_train_CountVec, y_train)
    X_test_cv = Count_Vectorizer.transform(X_test)
    y_pred = nb.predict(X_test_cv)
    with open('CountVec_NaiveBayes_Model.pkl', 'wb') as f:
        pickle.dump(nb, f)
    return calculateStatistics(y_pred)


def TFIDF_NaiveBayes(df):
    nb = MultinomialNB()
    nb.fit(X_train_TFIDF, y_train)
    X_test_tfidf = TFIDF_Vectorizer.transform(X_test)
    y_pred = nb.predict(X_test_tfidf)
    with open('TFIDF_NaiveBayes_Model.pkl', 'wb') as f:
        pickle.dump(nb, f)
    return calculateStatistics(y_pred)


def TFIDF_SVM(df):
    svm = LinearSVC()
    svm.fit(X_train_TFIDF, y_train)
    X_test_tfidf = TFIDF_Vectorizer.transform(X_test)
    y_pred = svm.predict(X_test_tfidf)
    with open('TFIDF_SVM_Model.pkl', 'wb') as f:
        pickle.dump(svm, f)
    return calculateStatistics(y_pred)


def CountVec_SVM(df):
    svm = LinearSVC()
    svm.fit(X_train_CountVec, y_train)
    X_test_cv = Count_Vectorizer.transform(X_test)
    y_pred = svm.predict(X_test_cv)
    with open('CountVec_SVM_Model.pkl', 'wb') as f:
        pickle.dump(svm, f)
    return calculateStatistics(y_pred)


def TFIDF_RF(df):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_TFIDF, y_train)
    X_test_tfidf = TFIDF_Vectorizer.transform(X_test)
    y_pred = rf.predict(X_test_tfidf)
    with open('TFIDF_RF_Model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    return calculateStatistics(y_pred)


def CountVec_RF(df):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_CountVec, y_train)
    X_test_cv = Count_Vectorizer.transform(X_test)
    y_pred = rf.predict(X_test_cv)
    with open('CountVec_RF_Model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    return calculateStatistics(y_pred)


def calculateStatistics(y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return [accuracy, precision, recall, f1]
