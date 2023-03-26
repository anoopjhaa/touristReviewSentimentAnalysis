# flask --app app.py --debug run

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from models import initializeModels, getVectorizers, getStatistics

app = Flask(__name__)

df = pd.read_csv('Reviews_balanced-10000.csv')
initializeModels(df)
clf = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    feature_extraction = request.form['feature_extraction']
    model = request.form['model']
    review = request.form['review']
    processed_review = None
    vectorizer = None

    if (feature_extraction == 'CountVec'):
        vectorizer = getVectorizers()[0]
        processed_review = CountVectorizer().fit_transform([review])
        if (model == 'NaiveBayes'):
            importModelFromPickleDump('CountVec_NaiveBayes_Model.pkl')
        elif (model == 'SVM'):
            importModelFromPickleDump('CountVec_SVM_Model.pkl')
        else:
            importModelFromPickleDump('CountVec_RF_Model.pkl')
    else:
        processed_review = TfidfVectorizer().fit_transform([review])
        vectorizer = getVectorizers()[1]
        if (model == 'NaiveBayes'):
            importModelFromPickleDump('TFIDF_NaiveBayes_Model.pkl')
        elif (model == 'SVM'):
            importModelFromPickleDump('TFIDF_SVM_Model.pkl')
        else:
            importModelFromPickleDump('TFIDF_RF_Model.pkl')

    review_vector = vectorizer.transform([review])
    rating_pred = clf.predict(review_vector)
    print(rating_pred[0])
    return render_template('index.html', result=rating_pred[0])


def importModelFromPickleDump(modelName):
    with open(modelName, 'rb') as f:
        global clf
        clf = pickle.load(f)


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    preprocessed_text = ' '.join(lemmas)
    return preprocessed_text


@app.route('/feedback', methods=['POST'])
def feedback():
    return render_template('index.html', feedback=True)


@app.route('/statistics')
def statistics():
    print(getStatistics())
    return render_template('statistics.html', data=getStatistics())


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/places')
def places():
    return render_template('places.html')


if __name__ == '__main__':
    app.run(debug=True)
