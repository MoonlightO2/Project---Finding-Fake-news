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

# Code reference: https://github.com/Spidy20/Fake_News_Detection/blob/master/Fake_News_Det.py

# Initialize Flask application
app = Flask(__name__)

# Creates an instance of the TfidfVectorizer class from scikit-learn
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)

# Loads a pre-trained machine learning model
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Reads a CSV file containing news data into a Pandas dataframe
dataframe = pd.read_csv('news.csv')

# Extracts the "text" column of the CSV file from the Pandas dataframe and assigns it to the variable x
x = dataframe['text']

# Extracts the "label" column of the CSV file from the Pandas dataframe and assigns it to the variable y
y = dataframe['label']

# Splits the data into training and testing sets using the train_test_split function from scikit-learn
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Code reference: https://github.com/Spidy20/Fake_News_Detection/blob/master/Fake_News_Det.py

# Takes in a news article and predicts whether it's fake or not
def fake_news_det(news):

    # Convert the input news article to a Pandas Series and fill any missing values with an empty string
    news = pd.Series(news).fillna('')

    # Fit and transform the training data using a TfidfVectorizer
    tfidf_x_train = tfvect.fit_transform(x_train.values.astype('U'))

    # Transform the test data using the same TfidfVectorizer
    tfidf_x_test = tfvect.transform(x_test.values.astype('U'))

    # Convert the input news article to a list and then transform it using the TfidfVectorizer
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data).str.lower()

    # Use a loaded machine learning model to predict whether the input news article is fake or not
    prediction = loaded_model.predict(vectorized_input_data)

    # Return the prediction
    return prediction

# This route handles GET and POST requests to '/' index / home
@app.route('/', methods=['GET', 'POST'])

#home() function returns a rendered HTML template called index.html
def home():
    return render_template('index.html')

# Code reference: https://github.com/Spidy20/Fake_News_Detection/blob/master/Fake_News_Det.py

# This route handles GET and POST requests to '/predict'
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    # Check if the incoming request is a POST request
    if request.method == 'POST':

        # Check if the 'message' field is present in the form data submitted with the POST request
        if request.form.get('message') is not None:

            # Extract the 'message' from the form data
            msg = request.form['message']

            # Call the 'fake_news_det' function to get the prediction for the input 'message'
            pred = fake_news_det(msg)

            # Print the prediction to the console for debugging purposes
            print(pred)

            # Render the 'prediction.html' template with the prediction value passed as a parameter
            return render_template('prediction.html', prediction=pred)
        else:

            # Render the 'prediction.html' template with an error message if the 'message' field is missing in the form data
            return render_template('prediction.html', prediction="Something went wrong")
    else:

        # Render the 'prediction.html' template with an error message if the incoming request is not a POST request
        return render_template('prediction.html', prediction="Something went wrong")


# This route handles GET and POST requests to '/random_news'
@app.route('/random_news', methods=['GET', 'POST'])
def random_news():

    # Opens a CSV file called 'news.csv' in read mode with UTF8 encoding
    with open('news.csv', 'r', encoding='UTF8') as f:

        # Creates a CSV reader object from the file
        reader = csv.DictReader(f)

        # Reads all the rows in the CSV file and stores them in a list
        rows = list(reader)

        # Selects a random row from the list of rows
        random_row = random.choice(rows)

        # Renders a template called 'random_news.html' and passes in the randomly selected row
        return render_template('random_news.html', row=random_row)


# This route handles GET and POST requests to '/process_form'
@app.route('/process_form', methods=['GET', 'POST'])
def process_form():
    # Retrieves the value of the form field named "message" from the request object
    message = request.form.get('message')

    # Calls a function named fake_news_det() and passes it the message variable as an argument
    pred = fake_news_det(message)

    # Prints the pred value to the console
    print(pred)

    # Returns a rendered template named "processed.html"
    return render_template('processed.html', prediction=pred)


# This function takes an article as input and returns an accuracy score
def classify_news(article):

    # Set the accuracy score to 0.85 (This value is hard-coded and does not change)
    accuracy_score = 0.85

    # Return the accuracy score
    return accuracy_score


# This route handles GET and POST requests to '/upload_file'
@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():

    # Render the 'upload.html' template and returns response
    return render_template('upload.html')


# This route handles GET and POST requests to '/accuracy'
@app.route('/accuracy', methods=['GET', 'POST'])
def accuracy():

    # Get the uploaded CSV file from the request
    file = request.files['csvfile']

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file)

    # Split the DataFrame into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # Create a CountVectorizer object to convert text data into numerical features
    vectorizer = CountVectorizer(stop_words='english')

    # Fit the vectorizer to the training data and transform both training and testing data
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Create a Multinomial Naive Bayes model and fit it to the training data
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Use the trained model to predict the labels for the testing data
    y_pred = model.predict(X_test)

    # Compute the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)

    # Render the 'accuracy.html' template and returns accuracy value
    return render_template('accuracy.html', accuracy=accuracy)


# Checks whether this file is being run directly, as opposed to being imported as a module.
if __name__ == '__main__':

    # This line starts the Flask web application with the following options:
    # - host: the IP address that the application should listen on (in this case, 127.0.0.1 or "localhost")
    # - port: the port number that the application should listen on (in this case, 8000)
    # - debug: whether to run the application in debug mode, which provides additional error information
    app.run(host="127.0.0.1", port=8000, debug=True)
