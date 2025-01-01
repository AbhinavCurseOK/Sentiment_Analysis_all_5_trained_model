from flask import Flask, render_template, request, redirect, url_for
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

models = {
    1: {'model': joblib.load('rf_classifier_model.joblib'), 'name': 'Random Forest Classifier'},
    2: {'model': joblib.load('logreg_model.pkl'), 'name': 'Logistic Regression'},
    3: {'model': joblib.load('TWT_svc_model.pkl'), 'name': 'Linear SVC'},
    4: {'model': joblib.load('best_linear_svc_model.pkl'), 'name': 'Linear SVC_1'},
    5: {'model': joblib.load('best_logreg_model_1.joblib'), 'name': 'Logistic Regression_1'}
}

vectorizer = joblib.load('tfidf_vectorizer.joblib')


def clean_text(text):
    """Cleans text using regular expressions."""
    patterns = [
        r'@[^ ]+',              
        r'https?://[A-Za-z0-9./]+', 
        r"\'s",                   
        r'\#\w+',                 
        r'&amp',                
        r'[^A-Za-z\s]'           
    ]
    combined_pattern = '|'.join(patterns)
    cleaned_text = re.sub(combined_pattern, '', text).lower().strip()
    return cleaned_text


def preprocess_text(text):
    """Tokenizes, lemmatizes, and removes stopwords."""
    default_stop_words = set(stopwords.words('english'))
    negation_words = {
        "no", "nor", "not", "don't", "aren't", "couldn't", "didn't",
        "doesn't", "hadn't", "hasn't", "haven't", "isn't", "mightn't",
        "mustn't", "needn't", "shan't", "shouldn't", "wasn't", "weren't",
        "won't", "wouldn't"
    }
    custom_stop_words = default_stop_words - negation_words
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in custom_stop_words]
    return " ".join(processed_tokens)


def sentiment_label(model_output):
    """Maps model outputs to sentiment labels."""
    if model_output == -1:
        return "Negative"
    elif model_output == 0:
        return "Neutral"
    elif model_output == 1:
        return "Positive"
    return "Unknown"


@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles text submission and prediction."""
    try:
        user_text = request.form.get('user_text', '')
        selected_model = int(request.form.get('model_number', 0))

        if not user_text.strip():
            return render_template('index.html', error="Please enter valid text.")

        if selected_model not in models:
            return render_template('index.html', error="Invalid model selected.")

        cleaned_text = clean_text(user_text)
        processed_text = preprocess_text(cleaned_text)

        vectorized_data = vectorizer.transform([processed_text])

        model_info = models[selected_model]
        model = model_info['model']
        model_name = model_info['name']

        prediction = model.predict(vectorized_data)
        sentiment = sentiment_label(model_output=prediction[0])

        return render_template('prediction.html', prediction=sentiment, model_name=model_name, user_text=user_text)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")


@app.route('/exit', methods=['POST', 'GET'])
def exit():
    """Redirects to the home page."""
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
