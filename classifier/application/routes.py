from flask import Flask, request, jsonify
from application import app 
from spam_classifier import Classifier
import pandas as pd

@app.route('/') 
def hello_world():
    email1 = 'Развивай бизнес на дому с услугой "Безлимитный Интернет"' 
    email2 = 'Мы получили ваше сообщение о пропаже багажа и домашнего питомца в здании аэропорта Конечно нам жаль Но что мы можем с этим сделать'
    email3 = 'Перезвони по номеру +799999999 в течение 6 секунд и выиграй миллион рублей!'
    test_set = [email1, email2, email3]
    result_dict = dict()
    for email in test_set:
        result_dict[email] = cls.classify(email)
    result = '; '.join(str(key) + '   :   ' + str(value) for key, value in result_dict.items())
    return (result)

@app.route('/classify_text', methods=['POST'])
def classify_text():
    data = request.json
    text = data.get('text')
    if text is None:
        params = ', '.join(data.keys())
        return jsonify({'message': f'Parametr "{params}" is invalid'}), 400
    else:
        result = cls.classify(text)
        return jsonify({'result': result})



# часть для простого обучения из п. 2.3  - cls
SPAM = 'SPAM'
NOT_SPAM = 'NOT_SPAM'
train_data = [  
    ['Купите новое чистящее средство', SPAM],   
    ['Купи мою новую книгу', SPAM],  
    ['Подари себе новый телефон', SPAM],
    ['Добро пожаловать и купите новый телевизор', SPAM],
    ['Привет давно не виделись', NOT_SPAM], 
    ['Довезем до аэропорта из пригорода всего за 399 рублей', SPAM], 
    ['Добро пожаловать в Мой Круг', NOT_SPAM],  
    ['Я все еще жду документы', NOT_SPAM],  
    ['Приглашаем на конференцию Data Science', NOT_SPAM],
    ['Потерял твой телефон напомни', NOT_SPAM],
    ['Порадуй своего питомца новым костюмом', SPAM]
]

cls = Classifier()
cls.train(train_data)
