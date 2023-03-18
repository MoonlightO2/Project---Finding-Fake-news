from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask
import csv
import random

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    news = pd.Series(news).fillna('')
    tfidf_x_train = tfvect.fit_transform(x_train.values.astype('U'))
    tfidf_x_test = tfvect.transform(x_test.values.astype('U'))
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/random_news', methods=['GET', 'POST'])
def random_news():
    with open('news.csv', 'r', encoding='UTF8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        random_row = random.choice(rows)
        return render_template('random_news.html', row=random_row)


@app.route('/process_form', methods=['GET', 'POST'])
def process_form():
    message = request.form.get('message')
    pred = fake_news_det(message)
    print(pred)
    return render_template('processed.html', prediction=pred)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('prediction.html', prediction=pred)
    else:
        return render_template('prediction.html', prediction="Something went wrong")


def classify_news(article):
    accuracy_score = 0.85
    return accuracy_score


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    return render_template('upload.html')


@app.route('/accuracy', methods=['GET', 'POST'])
def accuracy():
    file = request.files['csvfile']
    df = pd.read_csv(file)
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    vectorizer = CountVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return render_template('accuracy.html', accuracy=accuracy)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000, debug=True)
