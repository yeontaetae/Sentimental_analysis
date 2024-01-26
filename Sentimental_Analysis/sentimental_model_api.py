from transformers import pipeline
from flask import Flask, request

app = Flask(__name__)

distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True
)

@app.route("/analyze", methods=['POST'])
def analysis():
    text = request.json["text"]
    result = distilled_student_sentiment_classifier(text)
    return result

@app.route("/statement", methods=['POST'])
def statement():
    text = request.json["text"]
    result = distilled_student_sentiment_classifier(text)

    positive_score = result[0][0]
    neutral_score = result[0][1]
    negative_score = result[0][2]

    list = [positive_score, neutral_score, negative_score]

    max_trait = max(list, key=lambda r : r["score"])

    return {"score" : max_trait["score"]}

