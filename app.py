import pickle
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Flask app - starting point of our project
app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(message):
    # Lowercase the text
    text = message.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]
    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    # Stemming
    y = [ps.stem(i) for i in y]
    return " ".join(y)  # Preprocessed text

def predict_spam(message):
    try:
        # Preprocess the message
        transformed_sms = transform_text(message)
        # Vectorize the processed message
        vector_input = tfidf.transform([transformed_sms])
        # Predict using the ML model
        result = model.predict(vector_input)[0]
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error"

@app.route('/')  # Homepage
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_sms = request.form['message']
        result = predict_spam(input_sms)
        return render_template('index.html', result=result)

if __name__ == '__main__':
    # Load the vectorizer and model
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    # Run the Flask app
    app.run(host='0.0.0.0')
