from flask import Flask, request, jsonify, Response
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import json

# Загрузка необходимых данных
nltk.download('punkt')
nltk.download('wordnet')
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Порог вероятности для перенаправления
THRESHOLD = 0.35

def preprocess(text):
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

def predict(text):
    text_processed = preprocess(text)
    text_vectorized = vectorizer.transform([text_processed])
    predicted_class = model.predict(text_vectorized)[0]
    predicted_proba = model.predict_proba(text_vectorized)[0]
    max_proba = np.max(predicted_proba)

    if max_proba < THRESHOLD:
        return "Я не знаю, передал запрос оператору"

    responses = {
        "forgot_password": "Чтобы восстановить пароль, следуйте инструкциям на странице восстановления.",
        "change_email": "Чтобы сменить почту, зайдите в настройки аккаунта.",
        "change_nickname": "Чтобы сменить никнейм, перейдите в настройки профиля.",
        "project_authors": "Джумалиев Тимур является автором проекта.",
        "available_commands": "Доступные команды: смена пароля, смена почты, смена никнейма, авторы проекта."
    }

    return responses.get(predicted_class, "Я не знаю, передал запрос оператору")

app = Flask(__name__)

@app.route('/prompt', methods=['GET'])
def propt():
    text = request.args.get('text')
    if not text:
        return jsonify({"answer": "Текст для нейронки не был предоставлен"}), 400

    response = predict(text)
    return Response(json.dumps({"answer": response}, ensure_ascii=False), content_type='application/json; charset=utf-8')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
