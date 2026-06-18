from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def home():
    user_data = ""
    stw = []
    sentiment = ""

    if request.method == 'POST':
        user_data = request.form['my_text']
        
        if user_data:
            word = set(stopwords.words("english"))
            data = user_data.lower()
            
            token = word_tokenize(data)
            stw = []

            for i in token:
                if i not in word and i.isalnum(): 
                    stw.append(i)

            clean_sentence = " ".join(stw)
            text_to_score = clean_sentence if clean_sentence else data
            score = sia.polarity_scores(text_to_score)['compound']

            if score >= 0.05:
                sentiment = "Positive"
            elif score <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

    return render_template('index.html', 
                           original_text=user_data, 
                           processed_tokens=stw, 
                           result=sentiment)

app = app